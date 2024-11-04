from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from .template_class import AA
from flamo import dsp, system
from flamo.functional import (
    db2mag,
    mag2db,
    get_magnitude,
    get_eigenvalues,
    WGN_reverb
)
from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer



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

