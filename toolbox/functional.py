import torch
from flamo.functional import db2mag, sosfreqz, bandpass_filter


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
        

def wgn_reverb(
        matrix_size: tuple = (1, 1), t60: float = 1.0, samplerate: int = 48000, device=None
    ) -> torch.Tensor:
    r"""
    Generates White-Gaussian-Noise-reverb impulse responses.

        **Arguments**:
            - **matrix_size** (tuple, optional): (output_channels, input_channels). Defaults to (1,1).
            - **t60** (float, optional): Reverberation time. Defaults to 1.0.
            - **samplerate** (int, optional): Sampling frequency. Defaults to 48000.
            - **nfft** (int, optional): Number of frequency bins. Defaults to 2**11.

        **Returns**:
            torch.Tensor: Matrix of WGN-reverb impulse responses.
    """
    # Number of samples
    n_samples = int(1.5 * t60 * samplerate)
    # White Guassian Noise
    noise = torch.randn(n_samples, *matrix_size, device=device)
    # Decay
    dr = t60 / torch.log(torch.tensor(1000, dtype=torch.float32, device=device))
    decay = torch.exp(-1 / dr * torch.linspace(0, t60, n_samples))
    decay = decay.view(-1, *(1,) * (len(matrix_size))).expand(-1, *matrix_size)
    # Decaying WGN
    IRs = torch.mul(noise, decay)
    # Go to frequency domain
    TFs = torch.fft.rfft(input=IRs, n=n_samples, dim=0)

    # Generate bandpass filter
    fc_left = torch.tensor([20], dtype=torch.float32, device=device)
    fc_right = torch.tensor([20000], dtype=torch.float32, device=device)
    g = torch.tensor([1], dtype=torch.float32, device=device)
    b, a = bandpass_filter(
        fc1=fc_left, fc2=fc_right, gain=g, fs=samplerate, device=device
    )
    sos = torch.cat((b.reshape(1, 3), a.reshape(1, 3)), dim=1)
    bp_H = sosfreqz(sos=sos, nfft=n_samples).squeeze()
    bp_H = bp_H.view(*bp_H.shape, *(1,) * (len(TFs.shape) - 1)).expand(*TFs.shape)

    # Apply bandpass filter
    TFs = torch.mul(TFs, bp_H)

    # Return to time domain
    IRs = torch.fft.irfft(input=TFs, n=n_samples, dim=0)

    # Normalize
    vec_norms = torch.linalg.vector_norm(IRs, ord=2, dim=(0))
    return IRs / vec_norms