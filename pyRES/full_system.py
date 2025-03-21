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
from pyRES.physical_room import PhRoom
from pyRES.virtual_room import VrRoom


# ==================================================================
# ================ REVERBERATION ENHANCEMENT SYSTEM ================]

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
            virtual_room: VrRoom=None,
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
        # TODO: do not receive n_S, n_M, n_L, n_A as arguments, check that n_L and n_M are the same for physical and virtual room, and take n_S and n_A from the physical room
        # TODO: same with fs, nfft and alias_decay_db, do not receive them as argument, check compatibility between physical and virtual room
        self.n_S = n_S
        self.n_M = n_M
        self.n_L = n_L
        self.n_A = n_A

        self.__H = physical_room

        # Virtual room
        if virtual_room is None:
            virtual_room = dsp.Gain(size=(n_L, n_M), nfft=self.nfft, requires_grad=True, alias_decay_db=alias_decay_db)
        self.__v_ML = virtual_room

        # System gain
        self.__G = dsp.parallelGain(size=(n_L,), nfft=self.nfft, alias_decay_db=alias_decay_db)
        self.set_G_to_GBI()

        # Optimization routine
        self.__opt = system.Shell(
            core=self.open_loop(),
            input_layer=system.Series(
                dsp.Transform(lambda x: x.diag_embed()),
                dsp.FFT(self.nfft)
            )
        )

    # ==================================================================================
    # ================================== FORWARD PATH ==================================

    # TODO: cancel this
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
    # ============================ PHYSICAL ROOM METHODS ===============================

    def get_h_SA(self) -> nn.Module:
        f"""
        Returns the physical-room module from sound sources to audience positions.

            **Returns**:
                nn.Module: sound-source-to-audience-positions
        """
        return self.__H.h_SA
    
    def get_h_SM(self) -> nn.Module:
        f"""
        Returns the physical-room module from sound sources to microphones.

            **Returns**:
                nn.Module: sound-source-to-microphones
        """
        return self.__H.h_SM

    def get_h_LM(self) -> nn.Module:
        f"""
        Returns the physical-room module from loudspeakers to microphones.

            **Returns**:
                nn.Module: loudspeaker-to-microphone impulse-response module.
        """
        return self.__H.h_LM
    
    def get_h_LA(self) -> nn.Module:
        f"""
        Returns the physical-room module from loudspeakers to audience positions.

            **Returns**:
                nn.Module: loudspeaker-to-audience-positions
        """
        return self.__H.h_LA
    
    # ==================================================================================
    # ============================= VIRTUAL ROOM METHODS ===============================

    def get_v_ML(self) -> nn.Module:
        f"""
        Returns the virtual room.

            **Returns**:
                nn.Module: virtual room.
        """
        return self.__v_ML
    
    def get_v_ML_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time and frequency responses of the virtual room.
        
            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses
        """

        # Generate virtual room
        v_ml = system.Shell(
            core=self.get_v_ML()
        )
        with torch.no_grad():
            # Get the virtual room time and frequency responses
            v_ml_ir = v_ml.get_time_response(fs=self.fs, identity=True)
            v_ml_fr = v_ml.get_freq_response(fs=self.fs, identity=True)

        return v_ml_ir, v_ml_fr
    
    # ==================================================================================
    # ============================== SYSTEM GAIN METHODS ===============================
    
    def get_G(self) -> nn.Module:
        r"""
        Returns the system gain value in linear scale.

            **Returns**:
                torch.Tensor: system gain value (linear scale).
        """
        return self.__G

    def set_G(self, g: float) -> None:
        r"""
        Sets the system gain value to a value in linear scale.

            **Args**:
                g (float): new system gain value (linear scale).
        """
        assert isinstance(g, torch.FloatTensor), "G must be a torch.FloatTensor."
        self.__G.assign_value(g*torch.ones(self.n_L))

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
    # ================================= FEEDBACK LOOP ==================================

    def open_loop(self)-> system.Series:
        r"""
        Generates the system open loop.

            **Returns**:
                system.Series: Series object instance implementing the system open loop.
        """
        modules = OrderedDict([
            ('V_ML', self.get_v_ML()),
            ('G', self.get_G()),
            ('H_LM', self.get_h_LM())
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
            core=self.open_loop()
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
    
    def closed_loop(self) -> system.Recursion:
        r"""
        Generates a Recursion object instance representing the closed-loop system.

            **Returns**:
                system.Recurion: Recursion object instance implementing the closed-loop.
        """
        feedforward = system.Series(self.get_v_ML(), self.get_G())
        feedback = self.get_h_LM()
        return system.Recursion(fF=feedforward, fB=feedback)
    
    def closed_loop_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time- and frequency-response matrices of the closed-loop.

            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses.
        """

        # Generate closed loop
        closed_loop = system.Shell(
            core=self.closed_loop()
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
        closed_loop = self.closed_loop(self.get_v_ML(), self.get_G(), self.get_h_LM())
        
        # Build the electroacoustic path
        ea_components = system.Series(OrderedDict([
            ('H_SM', self.get_h_SM()),
            ('FeedbackLoop', closed_loop),
            ('H_LA', self.get_h_LA())
        ]))
        ea_path = system.Shell(core=ea_components, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))
        
        # Build the natural path
        nat_path = system.Shell(core=self.get_h_SA(), input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))

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
                self.__set_opt_inputLayer(system.Series(
                    dsp.Transform(lambda x: x.diag_embed()),
                    dsp.FFT(self.nfft))
                )
                self.__opt = system.Shell(core=self.open_loop())
            case 'closed_loop':
                self.__set_opt_inputLayer(system.Series(
                    dsp.Transform(lambda x: x.diag_embed()),
                    dsp.FFT(self.nfft))
                )
                self.__opt = system.Shell(core=self.closed_loop())
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