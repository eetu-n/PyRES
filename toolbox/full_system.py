# IMPORTS
from collections import OrderedDict
import torch
import torch.nn as nn
from flamo import dsp, system
from flamo.functional import (
    get_magnitude,
    get_eigenvalues,
)

from .physical_room import PhRoom


# TEMPLATE CLASS DEFINITION
class AAES(nn.Module):
    """
    Template for Active Acoustic Enhancement System (AAES) model.
    """
    def __init__(
            self,
            n_S: int=1,
            n_M: int=1,
            n_L: int=1,
            n_A: int=1,
            fs: int=48000,
            nfft: int=2**11,
            ph_room: PhRoom=None,
            vi_room: nn.Module=None,
            alias_decay_db: float=0
        ):
        """
        Initialize the Active Acoustics (AA) model. Stores system parameters and RIRs.

        Args:
            n_S (int, optional): number of natural sound sources. Defaults to 1.
            n_M (int, optional): number of microphones. Defaults to 1.
            n_L (int, optional): number of loudspeakers. Defaults to 1.
            n_A (int, optional): number of audience positions. Defaults to 1.
            fs (int, optional): sampling frequency. Defaults to 48000.
            nfft (int, optional): number of frequency bins. Defaults to 2**11.
            ph_room_name (str, optional): name of the physical room. Defaults to None.
            V_ML (nn.Module, optional): virtual room dsp. Defaults to None.
            alias_decay_db (float, optional): anti-time-aliasing decay in dB. Defaults to 0.
        """
        nn.Module.__init__(self)

        # Processing resolution
        self.fs = fs
        self.nfft = nfft

        # Physical room
        self.__H = ph_room
        self.H_SM = dsp.Filter(size=(self.__H.rir_length, n_M, n_S), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_SM.assign_value(self.__H.get_scs_to_mcs())
        self.H_SA = dsp.Filter(size=(self.__H.rir_length, n_A, n_S), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_SA.assign_value(self.__H.get_scs_to_aud())
        self.H_LM = dsp.Filter(size=(self.__H.rir_length, n_M, n_L), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_LM.assign_value(self.__H.get_lds_to_mcs())
        self.H_LA = dsp.Filter(size=(self.__H.rir_length, n_A, n_L), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_LA.assign_value(self.__H.get_lds_to_aud())

        # Virtual room
        self.G = dsp.parallelGain(size=(n_L,), nfft=self.nfft, alias_decay_db=alias_decay_db)
        self.G.assign_value(torch.ones(n_L))
        if vi_room is None:
            vi_room = dsp.Matrix(size=(n_L, n_M), nfft=self.nfft, matrix_type="random", requires_grad=False, alias_decay_db=alias_decay_db)
        self.V_ML = OrderedDict([('Virtual_Room', vi_room)])

        # Open Loop
        self.F_MM = system.Shell(
            core=self.__open_loop(self.get_V_ML(), self.get_G(), self.H_LM),
            input_layer=nn.Sequential(dsp.Transform(lambda x: x.diag_embed()), dsp.FFT(self.nfft))
            )
        self.set_G_to_GBI()

    # ==================================================================================
    # ================================== FORWARD PATH ==================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes one iteration of the feedback loop.

            **Args**:
                x (torch.Tensor): input signal.

            **Returns**:
                torch.Tensor: Depending on the input, it can be the microphones signals or the feedback loop matrix.
            
            **Usage**:
                If x is a vector of unit impulses of size (_, n_M), the output is a vector of size (_, n_M) representing the microphones signals.
                If x is a diagonal matrix of unit impulses of size (_, n_M, n_M), the output is a matrix of size (_, n_M, n_M) representing the feedback loop matrix.
                The first dimension of vectors and matrices depends on input_layer and output_layer of the Shell instance self.F_MM.
        """
        return self.F_MM(x)
    
    # ==================================================================================
    # ============================== SYSTEM GAIN METHODS ===============================
    
    def get_G(self) -> nn.Module:
        r"""
        Return the general gain value in linear scale.

            **Returns**:
                torch.Tensor: general gain value (linear scale).
        """
        return self.G

    def set_G(self, g: float) -> None:
        r"""
        Set the general gain value in linear scale.

            **Args**:
                g (float): new general gain value (linear scale).
        """
        assert isinstance(g, torch.FloatTensor), "G must be a float."
        self.G.assign_value(g*torch.ones(self.n_L))

    def get_GBI(self) -> torch.Tensor:
        r"""
        Return the Gain Before Instability (GBI) value in linear scale.

            **Returns**:
                torch.Tensor: GBI value (linear scale).
        """
        # Save current G value
        g_current = self.G.param.data[0].clone()

        # Set G to 1
        self.set_G(1)

        # Compute the gain before instability
        maximum_eigenvalue = torch.max(get_magnitude(self.get_open_loop_eigenvalues()))
        gbi = torch.reciprocal(maximum_eigenvalue)

        # Restore G value
        self.set_G(g_current)

        return gbi
    
    def set_G_to_GBI(self) -> None:
        r"""
        Set the system general gain to match the current system GBI in linear scale.
        """
        # Compute the current gain before instability
        gbi = self.get_GBI()

        # Apply the current gain before instability to the system general gain module
        self.set_G(gbi)

    # ==================================================================================
    # ============================= VIRTUAL ROOM METHODS ===============================

    def get_V_ML(self) -> OrderedDict:
        """
        Return a copy of the Virtual-Room-modules parameters.

        Returns:
            OrderedDict: modules' parameters
        """
        return self.V_ML

    # ==================================================================================
    # ================================= FEEDBACK LOOP ==================================

    def __open_loop(self, v_ml: OrderedDict, g: nn.Module, h_lm: nn.Module)-> nn.Sequential:
        r"""
        Generate a Series object instance representing one iteration of the feedback loop.

            **Args**:
                - h_lm (nn.Module): Feedback paths from loudspeakers to microphones.
                - v_ml (OrderedDict): Virtual room components.
                - g (nn.Module): General gain.

            **Returns**:
                nn.Sequential: Series implementing one open-loop iteration.
        """
        loop = system.Series()
        loop.append(v_ml)
        loop.append(g)
        loop.append(h_lm)
        
        return loop

    def get_open_loop_eigenvalues(self) -> torch.Tensor:
        r"""
        Compute the eigenvalues of the open-loop matrix.

            **Returns**:
                torch.Tensor: eigenvalues.
        """
        with torch.no_grad():

            # Compute eigenvalues
            evs = get_eigenvalues(self.F_MM.get_freq_response(fs=self.fs, identity=True))

        return evs

    # ==================================================================================
    # =============================== SYSTEM SIMULATION ================================

    def __create_system(self) -> tuple[system.Shell, system.Shell]:
        """
        Create the full system's Natural and Electroacoustic paths.

        Returns:
            tuple[Shell, Shell]: Natural and Electroacoustic paths as Shell objects.
        """
        # Build digital signal processor
        processor = system.Series()
        processor.append(self.get_V_ML())
        processor.add_module('G', self.get_G())
        # Build closed feedback loop
        feedback_loop = system.Recursion(fF=processor, fB=self.H_LM)
        # Build the electroacoustic path
        ea_components = nn.Sequential(OrderedDict([
            ('H_SM', self.H_SM),
            ('FeedbackLoop', feedback_loop),
            ('H_LA', self.H_LA)
        ]))
        ea_path = system.Shell(core=ea_components, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))
        # Build the natural path
        nat_path = system.Shell(core=self.H_SA, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))
        return nat_path, ea_path
    
    def system_simulation(self) -> torch.Tensor:
        """
        Simulate the full system. Produces the system impulse response.

        Returns:
            torch.Tensor: system impulse response.
        """
        with torch.no_grad():
            # Generate the paths
            nat_path, ea_path = self.__create_system()
            # Compute system response
            y = nat_path.get_time_response() + ea_path.get_time_response()

        return y