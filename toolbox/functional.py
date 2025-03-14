import torch
from flamo.functional import db2mag


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

    b = torch.zeros(2, *resonance.shape[1:], len(resonance))
    a = torch.zeros(3, *resonance.shape[1:], len(resonance))

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

    gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db)) / nfft / 20)
    b_aa = torch.einsum('p, pijk -> pijk', (gamma ** torch.arange(0, 2, 1)), b)
    a_aa = torch.einsum('p, pijk -> pijk', (gamma ** torch.arange(0, 3, 1)), a)

    B = torch.fft.rfft(b_aa, nfft, dim=0)
    A = torch.fft.rfft(a_aa, nfft, dim=0)
    
    return torch.div(B, A).sum(dim=-1)


