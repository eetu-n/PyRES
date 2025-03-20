# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
from collections import OrderedDict
# Torch
import torch
import torch.nn as nn
# Flamo
from flamo import dsp, system
from flamo.functional import get_magnitude, get_eigenvalues
# PyRES
from physical_room import PhRoom


# ==================================================================
# ================ REVERBERATION ENHANCEMENT SYSTEM ================
# ========================= TEMPLATE CLASS =========================
class RES(nn.Module):
    r"""
    Template for the Reverberation Enhancement System (RES) model.

        **Stores**:
            - system parameters
            - physical-room data
            - virtual-room data

        **Functionalities**:
            - system gain control
            - virtual-room time- and frequency-response matrices computation
            - system open loop time- and frequency-response matrices and eigenvalues computation
            - system closed loop time- and frequency-response matrices computation
            - full system simulation
            - optimization routine control for virtual-room parameter learning
            - system state management
    """
    def __init__(
            self,
            physical_room: PhRoom,
            n_S: int=1,
            n_M: int=1,
            n_L: int=1,
            n_A: int=1,
            fs: int=48000,
            nfft: int=2**11,
            virtual_room: nn.Module=None,
            alias_decay_db: float=0
        ):
        r"""
        Initializes the Reverberation Enhancement System (RES).

            **Args**:
                - physical_room (PhRoom): physical room.
                - n_S (int, optional): number of natural sound sources. Defaults to 1.
                - n_M (int, optional): number of microphones. Defaults to 1.
                - n_L (int, optional): number of loudspeakers. Defaults to 1.
                - n_A (int, optional): number of audience positions. Defaults to 1.
                - fs (int, optional): sampling frequency. Defaults to 48000.
                - nfft (int, optional): number of frequency bins. Defaults to 2**11.
                - virtual_room (nn.Module, optional): virtual room. Defaults to None.
                - alias_decay_db (float, optional): anti-time-aliasing decay in dB. Defaults to 0.
        """
        super().__init__()

        # Processing parameters
        self.fs = fs
        self.nfft = nfft
        self.alias_decay_db = alias_decay_db

        # Physical room
        self.n_S = n_S
        self.n_M = n_M
        self.n_L = n_L
        self.n_A = n_A

        self.__H = physical_room
        self.H_SM = dsp.Filter(
            size=(self.__H.rir_length, self.n_M, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        self.H_SM.assign_value(self.__H.get_stg_to_mcs())
        self.H_SA = dsp.Filter(
            size=(self.__H.rir_length, self.n_A, self.n_S),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        self.H_SA.assign_value(self.__H.get_stg_to_aud())
        self.H_LM = dsp.Filter(
            size=(self.__H.rir_length, n_M, n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        self.H_LM.assign_value(self.__H.get_lds_to_mcs())
        self.H_LA = dsp.Filter(
            size=(self.__H.rir_length, n_A, n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        self.H_LA.assign_value(self.__H.get_lds_to_aud())

        # Virtual room
        if virtual_room is None:
            virtual_room = dsp.Gain(size=(n_L, n_M), nfft=self.nfft, requires_grad=True, alias_decay_db=alias_decay_db)
        self.V_ML = virtual_room

        # System gain
        self.G = dsp.parallelGain(size=(n_L,), nfft=self.nfft, alias_decay_db=alias_decay_db)
        self.set_G_to_GBI()

        # Optimization routine
        self.__opt = system.Shell(core=self.__open_loop())

    # ==================================================================================
    # ================================== FORWARD PATH ==================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes one iteration of the optimization routine.

            **Args**:
                x (torch.Tensor): input signal.

            **Returns**:
                torch.Tensor: output of the optimization routine.
        """
        return self.__opt(x)
    
    # ==================================================================================
    # ============================== SYSTEM GAIN METHODS ===============================
    
    def get_G(self) -> nn.Module:
        r"""
        Returns the system gain value in linear scale.

            **Returns**:
                torch.Tensor: system gain value (linear scale).
        """
        return self.G

    def set_G(self, g: float) -> None:
        r"""
        Sets the system gain value to a value in linear scale.

            **Args**:
                g (float): new system gain value (linear scale).
        """
        assert isinstance(g, torch.FloatTensor), "G must be a torch.FloatTensor."
        self.G.assign_value(g*torch.ones(self.n_L))

    def compute_GBI(self, criterion: str='eigenvalue_magnitude') -> torch.Tensor:
        r"""
        Returns the system Gain Before Instability (GBI) value in linear scale.

            **Args**:
                - criterion (str, optional): criterion to compute the GBI. Defaults to 'eigenvalue_magnitude'.

            **Returns**:
                torch.Tensor: GBI value (linear scale).
        """
        match criterion:

            case 'eigenvalue_magnitude':
                with torch.no_grad():
                    # Save current G value
                    g_current = self.get_G().param.data[0].clone().detach()

                    # Set G to 1
                    self.set_G(torch.tensor(1.00))

                    # Compute the gain before instability
                    maximum_eigenvalue = torch.max(get_magnitude(self.open_loop_eigenvalues()))
                    gbi = torch.reciprocal(maximum_eigenvalue)

                    # Restore G value
                    self.set_G(g_current)

            case 'eigenvalue_real_part':
                with torch.no_grad():
                    # Save current G value
                    g_current = self.get_G().param.data[0].clone().detach()

                    # Set G to 1
                    self.set_G(torch.tensor(1.00))

                    # Compute the gain before instability
                    maximum_eigenvalue = torch.max(torch.real(self.open_loop_eigenvalues()))
                    gbi = torch.reciprocal(maximum_eigenvalue)

                    # Restore G value
                    self.set_G(g_current)

            case _:
                raise ValueError(f"Criterion '{criterion}' not recognized.")

        return gbi
    
    def set_G_to_GBI(self) -> None:
        r"""
        Sets the system gain to match the current system GBI.
        """
        # Compute the current gain before instability
        gbi = self.compute_GBI()

        # Apply the current gain before instability to the system gain module
        self.set_G(gbi)

    # ==================================================================================
    # ============================= VIRTUAL ROOM METHODS ===============================

    def __get_V_ML(self) -> nn.Module:
        f"""
        Returns the virtual room.

            **Returns**:
                nn.Module: virtual room.
        """
        return self.V_ML
    
    def get_V_ML_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time and frequency responses of the virtual room.
        
            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses
        """

        # Generate virtual room
        v_ml = system.Shell(
            core = self.__get_V_ML()
        )
        with torch.no_grad():
            # Get the virtual room time and frequency responses
            v_ml_ir = v_ml.get_time_response(fs=self.fs, identity=True)
            v_ml_fr = v_ml.get_freq_response(fs=self.fs, identity=True)

        return v_ml_ir, v_ml_fr

    # ==================================================================================
    # ================================= FEEDBACK LOOP ==================================

    def __open_loop(self)-> system.Series:
        r"""
        Generates the system open loop.

            **Returns**:
                system.Series: Series object instance implementing the system open loop.
        """
        modules = OrderedDict([
            ('V_ML', self.__get_V_ML()),
            ('G', self.get_G()),
            ('H_LM', self.H_LM)
        ])

        return system.Series(modules)
    
    def open_loop_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time- and frequency-response matrices of the open-loop.

            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses.
        """

        # Generate open loop
        open_loop = system.Shell(
            self.__open_loop(self.__get_V_ML(), self.get_G(), self.H_LM)
        )
        
        with torch.no_grad():
            # Compute open-loop time and frequency responses
            open_loop_irs = open_loop.get_time_response(fs=self.fs, identity=True)
            open_loop_fr = open_loop.get_freq_response(fs=self.fs, identity=True)

        return open_loop_irs, open_loop_fr

    def open_loop_eigenvalues(self) -> torch.Tensor:
        r"""
        Computes the eigenvalues of the system open-loop.

            **Returns**:
                torch.Tensor: open-loop eigenvalues.
        """

        # Generate open loop frequency responses
        _, fr_matrix = self.open_loop_responses()
        with torch.no_grad():
            # Compute eigenvalues
            evs = get_eigenvalues(fr_matrix)

        return evs
    
    def __closed_loop(self) -> system.Recursion:
        r"""
        Generate a Recursion object instance representing the closed-loop system.

            **Returns**:
                system.Recurion: Recursion object instance implementing the closed-loop.
        """
        feedforward = system.Series(self.__get_V_ML(), self.get_G())
        feedback = self.H_LM
        return system.Recursion(fF=feedforward, fB=feedback)
    
    def closed_loop_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the time- and frequency-response matrices of the closed-loop.

            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses.
        """

        # Generate closed loop
        closed_loop = system.Shell(
            core = self.__closed_loop(self.__get_V_ML(), self.get_G(), self.H_LM)
        )
        
        with torch.no_grad():
            # Compute closed-loop time and frequency responses
            closed_loop_irs = closed_loop.get_time_response(fs=self.fs, identity=True)
            closed_loop_fr = closed_loop.get_freq_response(fs=self.fs, identity=True)

        return closed_loop_irs, closed_loop_fr
    
    # ==================================================================================
    # =============================== SYSTEM SIMULATION ================================

    def __system_paths(self) -> tuple[system.Shell, system.Shell]:
        r"""
        Creates the full system's Natural and Electroacoustic paths.

            **Returns**:
                tuple[Shell, Shell]: Shell object instances implementing the natural and the electroacoustic paths of the RES.
        """
        # Build closed feedback loop
        closed_loop = self.__closed_loop(self.__get_V_ML(), self.get_G(), self.H_LM)
        
        # Build the electroacoustic path
        ea_components = system.Series(OrderedDict([
            ('H_SM', self.H_SM),
            ('FeedbackLoop', closed_loop),
            ('H_LA', self.H_LA)
        ]))
        ea_path = system.Shell(core=ea_components, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))
        
        # Build the natural path
        nat_path = system.Shell(core=self.H_SA, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))

        return nat_path, ea_path
    
    def system_simulation(self) -> torch.Tensor:
        r"""
        Simulates the full system producing the system impulse responses from the stage emitters to the audience receivers.

            **Returns**:
                torch.Tensor: system impulse response.
        """
        # Generate the paths
        nat_path, ea_path = self.__system_paths()
        
        with torch.no_grad():
            # Compute system response
            y = nat_path.get_time_response() + ea_path.get_time_response()

        return y
    
    # ==================================================================================
    # ============================= OPTIMIZATION ROUTINE ===============================

    def __set_opt_inputLayer(self, layer: nn.Module) -> None:
        r"""
        Sets the input layer of the optimization routine.

            **Args**:
                layer (nn.Module): input layer.
        """
        self.__opt.set_inputLayer(layer)

    def __set_opt_outputLayer(self, layer: nn.Module) -> None:
        r"""
        Sets the output layer of the optimization routine.

            **Args**:
                layer (nn.Module): output layer.
        """
        self.__opt.set_outputLayer(layer)

    def set_optimization_routine(self, to_optimize: str, input_layer: nn.Module=None, output_layer: nn.Module=None) -> None:
        r"""
        Sets the optimization routine.

            **Args**:
                - to_optimize (str): optimization routine.
                - input_layer (nn.Module, optional): input layer. Defaults to None.
                - output_layer (nn.Module, optional): output layer. Defaults to None.
        """
        match to_optimize:
            case 'open_loop':
                self.__opt = system.Shell(core=self.__open_loop())
            case 'closed_loop':
                self.__opt = system.Shell(core=self.__closed_loop())
            case _:
                raise ValueError(f"Optimization routine '{to_optimize}' not recognized.")
            
        if input_layer is not None:
            self.__set_opt_inputLayer(input_layer)
        if output_layer is not None:
            self.__set_opt_outputLayer(output_layer)
    
    # ==================================================================================
    # ================================= SYSTEM STATE ===================================

    def get_state(self) -> dict:
        r"""
        Returns the system current state.

            **Returns**:
                dict: model's state.
        """
        state = self.state_dict()
        return state
    
    def set_state(self, state: dict) -> None:
        r"""
        Sets the system current state.

            **Args**:
                state (dict): new state.
        """
        self.load_state_dict(state)

    def save_state(self, directory: str) -> None:
        r"""
        Saves the system current state.

            **Args**:
                directory (str): path to save the state.
        """
        torch.save(self.get_state(), directory)