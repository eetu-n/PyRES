from collections import OrderedDict

import torch
import torch.nn as nn

from flamo import dsp, system
from flamo.functional import (
    get_magnitude,
    get_eigenvalues,
)

from .room_loader import AA_RIRs

class AA(nn.Module):
    """
    Template for Active Acoustics (AA) model.
    """
    def __init__(self, n_S: int=1, n_M: int=1, n_L: int=1, n_A: int=1, fs: int=48000, nfft: int=2**11, room_name: str="Otala"):
        """
        Initialize the Active Acoustics (AA) model. Stores system parameters and RIRs.

        Args:
            n_S (int, optional): number of natural sound sources. Defaults to 1.
            n_M (int, optional): number of microphones. Defaults to 1.
            n_L (int, optional): number of loudspeakers. Defaults to 1.
            n_A (int, optional): number of audience positions. Defaults to 1.
            fs (int, optional): sampling frequency. Defaults to 48000.
            nfft (int, optional): number of frequency bins. Defaults to 2**11.
        """
        nn.Module.__init__(self)

        # Processing resolution
        self.fs = fs
        self.nfft = nfft

        # Sources, transducers, and audience
        self.n_S = n_S
        self.n_M = n_M
        self.n_L = n_L
        self.n_A = n_A

        # Physical room
        self.__Room = AA_RIRs(dir=room_name, n_S=self.n_S, n_L=self.n_L, n_M=self.n_M, n_A=self.n_A, fs=self.fs)
        self.__H_SM = dsp.Filter(size=(self.__Room.RIR_length, n_M, n_S), nfft=self.nfft, requires_grad=False)
        self.__H_SM.assign_value(self.__Room.get_scs_to_mcs())
        self.__H_SA = dsp.Filter(size=(self.__Room.RIR_length, n_A, n_S), nfft=self.nfft, requires_grad=False)
        self.__H_SA.assign_value(self.__Room.get_scs_to_aud())
        self.__H_LM = dsp.Filter(size=(self.__Room.RIR_length, n_M, n_L), nfft=self.nfft, requires_grad=False)
        self.__H_LM.assign_value(self.__Room.get_lds_to_mcs())
        self.__H_LA = dsp.Filter(size=(self.__Room.RIR_length, n_A, n_L), nfft=self.nfft, requires_grad=False)
        self.__H_LA.assign_value(self.__Room.get_lds_to_aud())

        # Virtual room
        self.__G = dsp.parallelGain(size=(self.n_L,))
        self.__G.assign_value(torch.ones(self.n_L))
        self.__V_ML = OrderedDict([('Virtual_Room', dsp.Gain(size=(self.n_L,self.n_M)))])

        # Feedback loop
        self.__GBI = dsp.parallelGain(size=(self.n_L,))
        self.__F_MM = system.Shell(model=self.__FL_iteration(self.__H_LM, self.__V_ML, self.__GBI))
        self.__update_GBI()

    # ----------------- FORWARD PATH ------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes one iteration of the feedback loop.

        Args:
            x (torch.Tensor): input signal

        Returns:
            torch.Tensor: Depending on the input, it can be the microphones signals or the feedback loop matrix.
        
        Usage:  if x is a vector of unit impulses of size (_, n_M), the output is a vector of size (_, n_M) representing the microphones signals.
                if x is a diagonal matrix of unit impulses of size (_, n_M, n_M), the output is a matrix of size (_, n_M, n_M) representing the feedback loop matrix.
                the first dimension of vectors and matrices depends on inputLayer and outputLayer of the Shell object self.__F_MM.
        """
        return self.__F_MM(x)
    
    # ----------------- OTHER METHODS -----------------
    # RIRs methods
    def get_RIRs(self) -> OrderedDict:
        """
        Returns a copy of the system RIRs.

        Returns:
            OrderedDict: System RIRs
        """
        RIRs = OrderedDict()
        RIRs.update({'H_SM': self.__H_SM.param.data.clone().detach()})
        RIRs.update({'H_SA': self.__H_SA.param.data.clone().detach()})
        RIRs.update({'H_LM': self.__H_LM.param.data.clone().detach()})
        RIRs.update({'H_LA': self.__H_LA.param.data.clone().detach()})
        return RIRs
    
    # General gain methods
    def get_G(self) -> float:
        """
        Returns the current general gain value.

        Returns:
            torch.Tensor: general gain value.
        """
        return self.__G.param.data.clone().detach()[0]

    def set_G(self, G: float) -> None:
        """
        Set the general gain value.

        Args:
            G (float): new general gain value.
        """
        assert isinstance(G, float), "G must be a float."
        with torch.no_grad():
            self.__G.assign_value(G*torch.ones(self.n_L))

    # Virtual Room methods
    def get_V_ML(self) -> OrderedDict: # NOTE: return only params or the full module?
        """
        Return a copy of the Virtual-Room-modules parameters.

        Returns:
            OrderedDict: modules' parameters
        """
        modules = OrderedDict()
        for _ in range(len(self.__V_ML)):
            key = next(iter(self.__V_ML))
            module = self.__V_ML[key]
            modules.update({key+'-'+str(type(module)): module.param.data.clone().detach()}) # NOTE: debug this line
        return modules

    def set_V_ML(self, V_ML: OrderedDict) -> None:
        """
        Set the Virtual-Room modules. Update feedback loop iteration matrix and GBI value.

        Args:
            V_ML (OrderedDict): New Virtual-Room modules.
        """
        assert isinstance(V_ML, OrderedDict), "V_ML must be an OrderedDict."
        with torch.no_grad():
            self.__V_ML = V_ML
            self.__update_FL()
            self.__update_GBI()

    def __save_V_ML_maps(self) -> OrderedDict:
        """
        Save the current Virtual-Room-modules maps.

        Returns:
            OrderedDict: current maps.
        """
        maps = OrderedDict()
        for key, module in self.__V_ML.items():
            maps.update({key: module.map})
        return maps

    def __normalize_V_ML(self) -> None: # NOTE: this is a wrong: if a filter has params != IRs then this function breaks the filter
                                        #       The two commented functions below will substitute these three once the DSP.map is a Parameterization
        """
        substitutes the current maps of the Virtual-Room-modules with a normalization map.
        """
        for value in self.__V_ML.values():
            value.map = lambda x: x/torch.norm(x, 'fro')

    def __restore_V_ML_maps(self, maps: OrderedDict) -> None:
        """
        Restore the Virtual-Room-modules maps.

        Args:
            maps (OrderedDict): maps to be restored.
        """
        for key,map in maps.items():
            self.__V_ML[key].map = map

    # def __add_V_ML_normalization(self) -> None: # NOTE: These two functions are really ugly, but wait for DSP.map to become Parameterization
    #     """
    #     Add normalization as last mapping step in each Virtual-Room module.
    #     """
    #     for value in self.__V_ML.values():
    #         map = value.map
    #         if not isinstance(map, nn.Sequential) and not isinstance(map, Transform):
    #             if callable(map) and map.__name__ == '<lambda>':
    #                 map = Transform(map)
    #             value.map = nn.Sequential(map, Transform(lambda x: x/torch.norm(x, 'fro')))
    #         else:
    #             value.map.add_module('Normalization', Transform(lambda x: x/torch.norm(x, 'fro')))

    # def __remove_V_ML_normalization(self) -> None:
    #     """
    #     Remove normalization as last mapping step in each Virtual-Room module.
    #     """
    #     for value in self.__V_ML.values():
    #         value.map = value.map[:-1]
    #         if len(value.map) == 1:
    #             value.map = value.map[0]

    # Feedback-loop matrix methods
    def __FL_iteration(self, H_LM: nn.Module, V_ML: OrderedDict, G: nn.Module)-> nn.Sequential:
        """
        Generate the feedback-loop iteration sequential.

        Args:
            H_LM (nn.Module): Feedback paths from loudspeakers to microphones.
            V_ML (OrderedDict): Virtual room components.
            G (nn.Module): Gain.

        Returns:
            nn.Sequential: Sequential implementing one feedback-loop iteration.
        """
        F_MM = nn.Sequential()
        for key,value in V_ML.items():
            F_MM.add_module(key, value)

        F_MM.add_module('GBI', G)
        F_MM.add_module('H_LM', H_LM)
        
        return F_MM
    
    def set_FL_inputLayer(self, layer: nn.Module) -> None:
        """
        Replaces the current feedback-loop input layer inside the Shell object.

        Args:
            layer (nn.Module): New input layer.
        """
        self.__F_MM.set_inputLayer(layer)
    
    def set_FL_outputLayer(self, layer: nn.Module) -> None:
        """
        Replaces the current feedback-loop output layer inside the Shell object.

        Args:
            layer (nn.Module): New output layer.
        """
        self.__F_MM.set_outputLayer(layer)
    
    def __update_FL(self) -> None:
        """
        Update the feedback loop components.
        """
        self.__F_MM.set_model(self.__FL_iteration(self.__H_LM, self.__V_ML, self.__GBI))

    def get_F_MM_eigenvalues(self) -> torch.Tensor:
        """
        Compute the eigenvalues of the feedback-loop matrix.

        Returns:
            torch.Tensor: eigenvalues.
        """
        with torch.no_grad():

            # Compute eigenvalues
            evs = get_eigenvalues(self.F_MM.get_freq_response(fs=self.fs, identity=True))

        return evs
    
    def __update_GBI(self) -> None:
        """
        Updates the loop normalizer gain to match the current system GBI
        """
        # compute GBI value
        maximum_eigenvalue = torch.max(get_magnitude(self.get_F_MM_eigenvalues()))
        GBI = 1 / maximum_eigenvalue
        # update GBI
        self.__GBI.assign_value(GBI * torch.ones(self.n_L))

    # Full system methods
    def __create_system(self) -> tuple[system.Shell, system.Shell]:
        """
        Create the full system's Natural and Electroacoustic paths.

        Returns:
            tuple[Shell, Shell]: Natural and Electroacoustic paths as Shell objects.
        """
        # Build digital signal processor
        processor = nn.Sequential()
        for key,value in self.V_ML.items():
            processor.add_module(key, value)
        processor.add_module('G', self.G)
        # Build feedback loop
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
    
    def system_simulation(self) -> torch.Tensor: # TODO: if I have n_S > 1 and/or n_A > 1 return all the IRs separated
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