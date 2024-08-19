from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torchaudio
import scipy.signal as sg
import matplotlib.pyplot as plt
import seaborn as sns

from .modules import (
    Transform,
    Gain,
    parallelGain,
    Filter,
    parallelFilter,
    Recursion,
    Shell
)
from .dsp import (
    FFT,
    iFFT,
)
from .utils.dataset import (
    Dataset
)
from .functional import (
    get_magnitude,
    get_eigenvalues,
    signal_gallery,
)
from .filters.functional import (
    db2mag, 
    bandpass_filter,
    sosfreqz
)


# ============================= Auxiliary ================================

class AA_RIRs(object): # NOTE: this is currently tailored for otala. When I have the sofa files I have to 
    def __init__(self, dir: str, n_S: int=1, n_L: int=1, n_M: int=1, n_A: int=1, fs: int=48000) -> None:
        """
        Room impulse response wrapper class.

        Args:
            dir (str): Path to the room impulse responses.
            n_S (int, optional): Number of sources. Defaults to 1.
            n_L (int, optional): Number of loudspeakers. Defaults to 1.
            n_M (int, optional): Number of microphones. Defaults to 1.
            n_A (int, optional): Number of audience members. Defaults to 1.
            fs (int, optional): Sample rate [Hz]. Defaults to 48000.
        """
        object.__init__(self)
        self.n_S = n_S
        self.n_L = n_L
        self.n_M = n_M
        self.n_A = n_A
        self.fs = fs
        self.dir = dir
        self.__RIRs = self.__load_rirs()
        self.RIR_length = self.__RIRs.shape[0]

    def __load_rirs(self) -> torch.Tensor:
        """
        Give the directory, loads the corresponding RIRs.

        Returns:
            torch.Tensor: RIRs. dtype=torch.float32, shape = (48000, n_M, n_L).
        """

        rirs = torch.zeros(48000, 5, 13)

        lds_index = [1,2,3,5,6,7,8,9,10,11,12,13,14]
        m = list(range(1,5+1))
        l = lds_index[0:13]

        for mcs in range(5):        # TODO: parametrize based on the room
            for lds in range(13):
                w, sr = torchaudio.load(f"./rirs/{self.dir}/mic{m[mcs]}_speaker{l[lds]}.wav")
                rirs[:,mcs,lds] = w.squeeze()

        if self.fs != sr:
            rirs = torchaudio.transforms.Resample(sr, self.fs)(rirs)

        rirs[:,1,:] = rirs[:,1,:] * db2mag(6)   # TODO: retake measurements, solve gain problems
        rirs[:,3,:] = rirs[:,3,:] * db2mag(-2)

        return rirs/(torch.norm(rirs, 'fro'))   # TODO: choose if and how to normalize

    def get_scs_to_mcs(self) -> torch.Tensor:
        """
        Returns the sources to microphones RIRs
        """
        return self.__RIRs[:, 0:self.n_M, 2].unsqueeze(2)

    def get_scs_to_aud(self) -> torch.Tensor:
        """
        Returns the sources to audience RIRs
        """
        return self.__RIRs[:, -1, 2].unsqueeze(1).unsqueeze(2)

    def get_lds_to_mcs(self) -> torch.Tensor:
        """
        Returns the loudspeakers to microphones RIRs
        """
        return self.__RIRs[:, 0:self.n_M, 0:self.n_L]

    def get_lds_to_aud(self) -> torch.Tensor:
        """
        Returns the loudspeakers to audience RIRs
        """
        return self.__RIRs[:, -1, 0:self.n_L].unsqueeze(1)


class MSE_evs(nn.Module):
    """
    Mean Squared Error (MSE) loss function for eigenvalues.
    To reduce computational complexity, the loss is computed only on a subset of the eigenvalues at each iteration of an epoch.
    """
    def __init__(self, ds_size, freq_points):
        """
        Initialize the MSE loss function for eigenvalues.

        Args:
            ds_size (_type_): _description_
            freq_points (_type_): _description_
        """
        super().__init__()
        # The number of intervals matches the dataset size
        self.interval_idxs = torch.randperm(ds_size)
        # The number of eigenvalues common to two adjacent intervals
        self.overlap = torch.tensor(500, dtype=torch.int)
        # The number of eigenvalues per interval
        int_width = torch.max(torch.tensor([freq_points//ds_size, 2400], dtype=torch.int))
        self.evs_numb = torch.tensor(int_width + self.overlap, dtype=torch.int)
        assert self.evs_numb < freq_points, "The number of eigenvalues per interval is too large."
        # Auxiliary variable to go through the intervals
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        # Get the indexes of the current interval
        idx1, idx2 = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idx1:idx2,:,:]))
        evs_true = y_true[:,idx1:idx2,:]
        # Compute the loss
        MSE = torch.mean(torch.square(evs_pred - evs_true))
        return MSE

    def __get_indexes(self):
        # Don't exceed the size of the tensors
        max_index = self.evs_numb * len(self.int_idx)
        min_index = 0
        # Compute indeces
        idx1 = torch.max(torch.tensor( [min_index, self.int_idx[self.i]*self.evs_numb - self.overlap], dtype=torch.int))
        idx2 = torch.min(torch.tensor( [(self.int_idx[self.i]+1)*self.evs_numb - self.overlap, max_index], dtype=torch.int))
        # Update interval counter
        self.i = (self.i+1) % len(self.int_idx)
        return idx1, idx2


# ==================== Active Acoustics Core Class =======================

class AA(nn.Module):
    """
    Template for Active Acoustics (AA) model.
    """
    def __init__(self, n_S: int=1, n_M: int=1, n_L: int=1, n_A: int=1, fs: int=48000, nfft: int=2**11, room_dir: str="Otala-2024.05.10"):
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
        self.__Room = AA_RIRs(dir=room_dir, n_S=self.n_S, n_L=self.n_L, n_M=self.n_M, n_A=self.n_A, fs=self.fs)
        self.__H_SM = Filter(size=(self.__Room.RIR_length, n_M, n_S), nfft=self.nfft, requires_grad=False)
        self.__H_SM.assign_value(self.__Room.get_scs_to_mcs())
        self.__H_SA = Filter(size=(self.__Room.RIR_length, n_A, n_S), nfft=self.nfft, requires_grad=False)
        self.__H_SA.assign_value(self.__Room.get_scs_to_aud())
        self.__H_LM = Filter(size=(self.__Room.RIR_length, n_M, n_L), nfft=self.nfft, requires_grad=False)
        self.__H_LM.assign_value(self.__Room.get_lds_to_mcs())
        self.__H_LA = Filter(size=(self.__Room.RIR_length, n_A, n_L), nfft=self.nfft, requires_grad=False)
        self.__H_LA.assign_value(self.__Room.get_lds_to_aud())

        # Virtual room
        self.__G = parallelGain(size=(self.n_L,))
        self.__G.assign_value(torch.ones(self.n_L))
        self.__V_ML = OrderedDict([('Virtual_Room', Gain(size=(self.n_L,self.n_M)))])

        # Feedback loop
        self.__GBI = parallelGain(size=(self.n_L,))
        self.__F_MM = Shell(model=self.__FL_iteration(self.__H_LM, self.__V_ML, self.__GBI))
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
        # save state
        inputLayer_save = self.__F_MM.get_inputLayer()
        outputLayer_save = self.__F_MM.get_outputLayer()
        maps = self.__save_V_ML_maps()

        # Compute eigenvalues
        self.__normalize_V_ML()
        self.__F_MM.set_inputLayer(nn.Sequential(Transform(lambda x: x.diag_embed()), FFT(self.nfft)))
        self.__F_MM.set_outputLayer(nn.Sequential(Transform(get_eigenvalues)))
        x = signal_gallery(batch_size=1, n_samples=96000, n=self.n_M, signal_type='impulse', fs=self.fs)
        evs = self.__F_MM(x)

        # restore state
        self.__restore_V_ML_maps(maps)
        self.__F_MM.set_inputLayer(inputLayer_save)
        self.__F_MM.set_outputLayer(outputLayer_save)

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
    def __create_system(self) -> tuple[Shell, Shell]:
        """
        Create the full system's Natural and Electroacoustic paths.

        Returns:
            tuple[Shell, Shell]: Natural and Electroacoustic paths as Shell objects.
        """
        # Build dsp
        dsp = nn.Sequential()
        for key,value in self.__V_ML.items():
            dsp.add_module(key, value)
        dsp.add_module('G', self.__G)
        dsp.add_module('GBI', self.__GBI)
        # Build feedback loop
        FeedbackLoop = Recursion(dsp, self.__H_LM)
        # Build the electroacoustic path
        EA_components = nn.Sequential(OrderedDict([
            ('H_SM', self.__H_SM),
            ('FeedbackLoop', FeedbackLoop),
            ('H_LA', self.__H_LA)
        ]))
        EA_path = Shell(model=EA_components, inputLayer=FFT(self.nfft), outputLayer=iFFT(self.nfft))
        # Build the natural path
        Nat_path = Shell(model=self.__H_SA, inputLayer=FFT(self.nfft), outputLayer=iFFT(self.nfft))
        return Nat_path, EA_path
    
    def system_simulation(self) -> torch.Tensor: # TODO: if I have n_S > 1 and/or n_A > 1 return all the IRs separated
        """
        Simulate the full system. Produces the system impulse response.

        Returns:
            torch.Tensor: system impulse response.
        """
        # Input impulse signal
        x = signal_gallery(batch_size=1, n_samples=96000, n=self.n_S, signal_type='impulse', fs=self.fs)
        # Save dsp state
        maps = self.__save_V_ML_maps()
        # Normalize the dsp components
        self.__normalize_V_ML()
        # Generate the paths
        Nat_path, EA_path = self.__create_system()
        # Compute system response
        y = Nat_path(x) + EA_path(x)
        # Restore dsp state
        self.__restore_V_ML_maps(maps)
        return y


# ========================= Example ==========================

class AA_dafx24(AA):
    """
    Reference:
        De Bortoli G., Dal Santo G., Prawda K., Lokki T., Välimäki V., and Schlecht S. J.
        Differentiable Active Acoustics---Optimizing Stability via Gradient Descent
        Int. Conf. on Digital Audio Effects (DAFx), Sep. 2024
    """
    def __init__(self, n_S: int=1, n_M: int=1, n_L: int=1, n_A: int=1, fs: int=48000, nfft: int=2**11, room_dir: str="Otala-2024.05.10", FIR_order: int=100):
        AA.__init__(self, n_S=n_S, n_M=n_M, n_L=n_L, n_A=n_A, fs=fs, nfft=nfft, room_dir=room_dir)

        # Virtual room
        self.U = Filter(size=(FIR_order, self.n_L, self.n_M), nfft=self.nfft, requires_grad=True)
        V_ML = OrderedDict([ ('U', self.U) ])
        self.set_V_ML(V_ML)
        self.set_FL_inputLayer(nn.Sequential(Transform(lambda x: x.diag_embed()), FFT(self.nfft)))

    def add_WGN(self, RT: float=1.0) -> None:
        reverb_order = self.nfft
        if int(RT*self.fs) > reverb_order:
            warnings.warn(f"Desired RT exceeds nfft value. 60 dB decrease in reverb energy will not be reached.")
        self.R = parallelFilter(size=(reverb_order, self.n_L), nfft=self.nfft, requires_grad=False)
        self.R.assign_value(self.WGN_irs(matrix_size=(reverb_order, self.n_L), RT=RT, nfft=self.nfft))
        V_ML = OrderedDict([
            ('U', self.U),
            ('R', self.R)
        ])
        self.set_V_ML(V_ML)
        
    def WGN_irs(self, matrix_size: tuple=(1,1,1), RT: float=1.0, nfft: int=2**11) -> torch.Tensor:
        """
        Generate White-Gaussian-Noise-reverb impulse responses.

        Args:
            matrix_size (tuple, optional): (reverb_order, output_channels, input_channels). Defaults to (1,1,1).
            RT (float, optional): Reverberation time. Defaults to 1.0.
            nfft (int, optional): Number of frequency bins. Defaults to 2**11.

        Returns:
            torch.Tensor: Matrix of WGN-reverb impulse responses.
        """

        # White Guassian Noise
        noise = torch.randn(*matrix_size)
        # Decay
        dr = RT/torch.log(torch.tensor(1000, dtype=torch.float32))
        decay = torch.exp(-1/dr*torch.linspace(0, RT, matrix_size[0]))
        decay = decay.view(*decay.shape, *(1,)*(len(matrix_size)-1)).expand(*matrix_size)
        # Decaying WGN
        IRs = torch.mul(noise, decay)
        # Go to frequency domain
        TFs = torch.fft.rfft(input=IRs, n=nfft, dim=0)

        # Generate bandpass filter
        fc_left = 20
        fc_right = 20000
        b,a = bandpass_filter(fc_left, fc_right, self.fs)
        sos = torch.cat((b.expand(1,1,3), a.expand(1,1,3)), dim=2)
        bp_H = sosfreqz(sos=sos, nfft=nfft).squeeze()
        bp_H = bp_H.view(*bp_H.shape, *(1,)*(len(TFs.shape)-1)).expand(*TFs.shape)

        # Apply bandpass filter
        TFs = torch.mul(TFs, bp_H)

        # Return to time domain
        IRs = torch.fft.irfft(input=TFs, n=nfft, dim=0) # NOTE: this is a very good candidate for anti-time-aliasing debugging

        # Normalize
        vec_norms = torch.linalg.vector_norm(IRs, ord=2, dim=(0))
        return IRs / vec_norms
    
    def plot_evs(evs, *kwargs):
        """
        Plot the magnitude distribution of the given eigenvalues.

        Args:
            evs (_type_): _description_
        """
        plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
        plt.figure(figsize=(7,6))
        ax = plt.subplot(1,1,1)
        for i in range(evs.shape[2]):
            evst = torch.reshape(evs[:,:,:,i], (evs.shape[1]*evs.shape[2], -1)).squeeze()
            evst_max = torch.max(evst, 0)[0]
            sns.boxplot(evst.numpy(), positions=[i], width=0.7, showfliers=False)
            ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black')

        plt.xticks([0,1], ['Initialization', 'Optimized'])
        plt.xticks(rotation=90)
        ax.yaxis.grid(True)
        plt.tight_layout()

        plt.show()

def AA_dafx24_test(args):

    # Model
    FIR_order = 100
    model = AA_dafx24(
        n_S = 1,
        n_M = args.input_channels,
        n_L = args.output_channels,
        n_A = 1,
        fs = args.samplerate,
        nfft = args.nfft,
        FIR_order = FIR_order
        )
    reverb_RT = 0.5
    model.add_WGN(reverb_RT)
    
    # save initialization response
    with torch.no_grad():
        evs_init = get_magnitude(model.get_F_MM_eigenvalues())
        y_init = model.system_simulation()
    # save_audio(os.path.join(args.train_dir, 'y_init.wav'), y_init[0, :], args.samplerate)

    # Initialize dataset
    dataset = Dataset_Colorless(
        in_shape = (args.nfft//2+1, model.n_M),
        target_shape = (args.nfft//2+1, model.n_M),
        ds_len = args.num,
        device = args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=args.batch_size, split=args.split, shuffle=args.shuffle)

    # Initialize training process
    criterion = MSE_evs(args.num, args.nfft//2+1)
    trainer = Trainer(model, args.max_epochs, args.lr, args.device)
    trainer.register_criterion(criterion, 1)
    # Train the model
    trainer.train(train_loader, valid_loader)

    # save optimized response
    with torch.no_grad():
        evs_optim = get_magnitude(model.get_F_MM_eigenvalues())
        y_optim = model.system_simulation()
    save_audio(os.path.join(args.train_dir, 'y_out.wav'), y_optim[0, :], args.samplerate)

    plt.figure()
    plt.plot(mag2db(evs_init[0, :]).detach().numpy(), label='Initial')
    plt.plot(mag2db(evs_optim[0, :]).detach().numpy(), label='Optimized')
    plt.legend()

    plt.figure()
    plt.plot(y_init[0, :].detach().numpy(), label='Initial')
    plt.plot(y_optim[0, :].detach().numpy(), label='Optimized')
    plt.legend()

    plt.figure()
    plt.subplot(2,1,1)
    plt.specgram(y_init[0,:].detach().squeeze().numpy())
    plt.subplot(2,1,2)
    plt.specgram(y_optim[0,:].detach().squeeze().numpy())

    evs = torch.cat((evs_init.unsqueeze(3), evs_optim.unsqueeze(3)), (-1))

    model.plot_evs(mag2db(evs[:,20:20000,:,:]))
    return None
