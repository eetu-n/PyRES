import torch
import torch.nn as nn
from flamo import dsp
from flamo.utils import to_complex
from .functional import modal_reverb, wgn_reverb

class unitary_connections(dsp.Matrix):
    pass

class random_FIRs(dsp.Filter):
    """
    Reference:
        De Bortoli G., Dal Santo G., Prawda K., Lokki T., Välimäki V., and Schlecht S. J.
        "Differentiable Active Acoustics---Optimizing Stability via Gradient Descent"
        International Conference on Digital Audio Effects (DAFx), Sep. 2024, Guilford, UK.
    """
    def __init__(
            self,
            n_M: int=1,
            n_L: int=1,
            nfft: int=2**11,
            FIR_order: int=100
        ) -> None:
        
        super.__init__(self, size=(FIR_order, n_L, n_M), nfft=nfft, requires_grad=True)


class phase_canceling_modal_reverb(dsp.Filter):
    """
    Reference:
        De Bortoli G., Prawda K., and Schlecht S. J.
        "Active Acoustics With a Phase Cancelling Modal Reverberator"
        Journal of the Audio Engineering Society 72, no. 10 (2024): 705-715.
    """
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        fs: int = 48000,
        nfft: int = 2**11,
        n_modes: int=10,
        low_f_lim: float=0,
        high_f_lim: float=500,
        t60: float=1.0,
        requires_grad: bool=False,
        alias_decay_db: float=0.0,
    ):
        super().__init__(size=(n_modes, n_L, n_M), nfft=nfft, requires_grad=requires_grad, alias_decay_db=alias_decay_db)

        self.fs = fs
        self.n_modes = n_modes
        self.resonances = torch.linspace(low_f_lim, high_f_lim, n_modes).view(-1, *(1,)*(len(self.param.shape[1:]))).reshape(*(1,)*len(self.param.shape[1:]), -1).expand(*self.param.shape)
        self.gains = torch.ones_like(self.param)
        self.t60 = t60 * torch.ones_like(self.param)
        self.gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db)) / nfft / 20)
    
    def init_param(self):
        torch.nn.init.uniform_(self.param, a=0, b=2*torch.pi)

    def get_freq_response(self):

        phases = self.param.permute(*torch.roll(torch.arange(len(self.param.shape)),-1)).unsqueeze(0)
        self.freq_response = modal_reverb(fs=self.fs, nfft=self.nfft, f_res=self.freqs, gain=self.gains, phase=phases, t60=self.t60)

        if torch.isnan(self.freq_response).any():
            print("Warning: NaN values in the frequency response. This is a common issue with high order, we are working on it. But please rise an issue on github if you encounter it. One thing that can help is to reduce the learning rate.")

    # def initialize_class(self):
    #     self.check_param_shape()
    #     self.get_io()
    #     self.freq_response = to_complex(
    #         torch.empty((self.nfft // 2 + 1, *self.size[1:]))
    #     )
    #     self.get_freq_response()
    #     self.get_freq_convolve()


class unitary_allpass_reverb():
    pass


class WGN_reverb(dsp.DSP):
    # TODO: After modifying wgn_reverb in FLAMO, this class has RT and mean and/or std as parameters to be optimized
    # TODO: wgn_reverb is used in get_freq_response()
    def __init__(
        self,
        n_M: int=1,
        n_L: int=1,
        nfft: int=2**11,
        RT: float=1.0,
        mean: float=0.0,
        std: float=1.0,
        requires_grad: bool=False
    ):
        pass
    pass