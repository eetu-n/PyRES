import torch
from flamo import dsp, system
from flamo.functional import db2mag
from flamo.auxiliary.reverb import rt2absorption


def one_pole_absorption_filter(fs, delays, rt_DC, rt_NY):

    b = torch.zeros(3, *delays.shape[1:])
    a = torch.ones(3, *delays.shape[1:])

    mag_DC = rt2absorption(rt_DC, fs, delays)
    mag_NY = rt2absorption(rt_NY, fs, delays)

    r = mag_DC / mag_NY

    a1 = (1 - r) / (1 + r)
    b0 = (1 - a1) * mag_NY

    b[0,:] = b0
    a[1,:] = a1


class FDN_absorption(dsp.parallelFilter):
    """
    Reference:
        De Bortoli G., Prawda K., and Schlecht S. J.
        "Active Acoustics With a Phase Cancelling Modal Reverberator"
        Journal of the Audio Engineering Society 72, no. 10 (2024): 705-715.
    """
    def __init__(
        self,
        delays: torch.Tensor,
        fs: int = 48000,
        nfft: int = 2**11,
        t60_DC: float = 1.0,
        t60_NY: float = 1.0,
        alias_decay_db: float = 0.0,
    ):
        super().__init__(size=(1, len(delays)), nfft=nfft, requires_grad=False, alias_decay_db=alias_decay_db)

        self.assign_value(delays)
        self.fs = fs
        self.t60_DC = t60_DC
        self.t60_NY = t60_NY
        self.gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db)) / nfft / 20)
        self.initialize_class()

    def compute_freq_response(self, delays: torch.Tensor) -> torch.Tensor:

        b, a = one_pole_absorption_filter(self.fs, delays, self.t60_DC, self.t60_NY)

        b = b.unsqueeze(0)
        a = a.unsqueeze(0)

        b_aa = torch.einsum('p, p... -> p...', (self.gamma ** torch.arange(0, 2, 1)), b)
        a_aa = torch.einsum('p, p... -> p...', (self.gamma ** torch.arange(0, 3, 1)), a)

        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)

        return torch.div(B, A).sum(dim=-1)
    
    def get_freq_response(self, param):

        self.freq_response = self.compute_freq_response(param)

        if torch.isnan(self.freq_response).any():
            print("Warning: NaN values in the frequency response. This is a common issue with high order, we are working on it. But please rise an issue on github if you encounter it. One thing that can help is to reduce the learning rate.")
        

def resonance_filter(
        fs: int, resonance:torch.Tensor, gain:torch.Tensor, phase:torch.Tensor, t60:torch.Tensor
    ):
    r"""
    Given a sampling rate, a resonance frequency and a reverberation
    time, builds the corresponding complex first order resonance filter
    returning its transfer function coefficients.

        **Args**:
            - f_res (torch.Tensor): The resonance frequency of the mode in Hz.
            - gain (torch.Tensor): The magnitude peak values in dB.
            - phase (torch.Tensor): The phase of the resonance.
            - t60 (float): The reverberation time of the resonance in seconds.
            - fs (int, optional): The sampling frequency of the signal in Hz. Defaults to 48000.

        **Returns**:
            - b (torch.Tensor): The numerator coefficients of the filter transfer function.
            - a (torch.Tensor): The denominator coefficients of the filter transfer function.
    """

    assert resonance.shape == gain.shape == phase.shape == t60.shape, "The input tensors must have the same shape."

    b = torch.zeros(2, *resonance.shape[1:])
    a = torch.zeros(3, *resonance.shape[1:])

    # phase = phase.permute(*torch.roll(torch.arange(len(phase.shape)),-1)).unsqueeze(0)

    # f_res = f_res.view(-1, *(1,)*(len(phase.shape[1:]))).reshape(*(1,)*len(phase.shape[1:]), -1).expand(*phase.shape)
    # gain = gain.view(-1, *(1,)*(len(phase.shape[1:]))).reshape(*(1,)*len(phase.shape[1:]), -1).expand(*phase.shape)
    
    # Normalized resonance
    res_norm = 2*torch.pi*resonance/fs

    # Pole radius
    radius = db2mag(-60 / (t60 * fs))
    # Pole
    pole = radius * torch.exp(torch.complex(torch.zeros_like(res_norm), res_norm))

    # Residue gain
    g = (1 - radius) * gain
    # Residue
    residue = g * torch.exp(torch.complex(torch.zeros_like(phase), phase))
    
    # Rational transfer function coefficients
    b[0] = 2*residue.real
    b[1] = -2*torch.real(residue*torch.conj(pole))
    a[0] = 1
    a[1] = -2*pole.real
    a[2] = torch.abs(pole)**2

    return b, a


def modal_reverb(
        fs: int, nfft: int, resonances: torch.Tensor, gains: torch.Tensor, phases: torch.Tensor, t60: torch.Tensor, alias_decay_db: float
    ) -> torch.Tensor:

    assert resonances.shape == gains.shape == phases.shape == t60.shape, "The input tensors must have the same shape."

    b, a = resonance_filter(fs=fs, resonance=resonances, gain=gains, phase=phases, t60=t60)

    gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db)) / nfft / 20) # TODO: This should go inside the class
    # n -> time samples / i,j -> input-ouput / k -> number
    b_aa = torch.einsum('n, n... -> n...', (gamma ** torch.arange(0, 2, 1)), b)
    a_aa = torch.einsum('n, n... -> n...', (gamma ** torch.arange(0, 3, 1)), a)

    B = torch.fft.rfft(b_aa, nfft, dim=0)
    A = torch.fft.rfft(a_aa, nfft, dim=0)
    
    return torch.div(B, A).sum(dim=-1)
