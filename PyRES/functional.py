# ==================================================================
# ============================ IMPORTS =============================
# PyTorch
import torch
# FLAMO
from flamo import dsp
from flamo.functional import db2mag
from flamo.auxiliary.reverb import rt2slope


# ==================================================================

def resonance_filter(
        fs: int, resonance:torch.Tensor, gain:torch.Tensor, phase:torch.Tensor, t60:torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Returns the transfer function coefficients of a complex first order resonance filter.

        **Args**:
            - fs (int, optional): The sampling frequency of the signal [Hz].
            - resonance (torch.Tensor): The resonance frequency of the mode [Hz].
            - gain (torch.Tensor): The magnitude peak values [dB].
            - phase (torch.Tensor): The phase of the resonance [rad].
            - t60 (float): The reverberation time of the resonance [s].

        **Returns**:
            - b (torch.Tensor): The numerator coefficients of the filter transfer function.
            - a (torch.Tensor): The denominator coefficients of the filter transfer function.
    """

    assert resonance.shape == gain.shape == phase.shape == t60.shape, "The input tensors must have the same shape."

    if len(resonance.shape) > 1:
        b = torch.zeros(2, *resonance.shape[1:])
        a = torch.zeros(3, *resonance.shape[1:])
    else:
        b = torch.zeros(2,1)
        a = torch.zeros(3,1)

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
    r"""
    Returns the transfer function of the modal reverb.

        **Args**:
            - fs (int): The sampling frequency of the signal [Hz]].
            - nfft (int): FFT size.
            - resonances (torch.Tensor): The resonance frequencies of the modes [Hz].
            - gains (torch.Tensor): The magnitude peak values [dB].
            - phases (torch.Tensor): The phase of the resonances [rad].
            - t60 (float): The reverberation time of the modal reverb [s].
            - alias_decay_db (float): The anti-time-aliasing decay [dB].

        **Returns**:
            - torch.Tensor: The transfer function of the modal reverb.
    """

    assert resonances.shape == gains.shape == phases.shape == t60.shape, "The input tensors must have the same shape."

    b, a = resonance_filter(fs=fs, resonance=resonances, gain=gains, phase=phases, t60=t60)

    gamma = 10 ** (-torch.abs(torch.tensor(alias_decay_db)) / nfft / 20) # TODO: This should go inside the class
    # n -> time samples / i,j -> input-ouput / k -> number
    b_aa = torch.einsum('n, n... -> n...', (gamma ** torch.arange(0, 2, 1)), b)
    a_aa = torch.einsum('n, n... -> n...', (gamma ** torch.arange(0, 3, 1)), a)

    B = torch.fft.rfft(b_aa, nfft, dim=0)
    A = torch.fft.rfft(a_aa, nfft, dim=0)
    
    return torch.div(B, A).sum(dim=-1)


def one_pole_filter(mag_DC: float, mag_NY: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Returns the coefficients of a one-pole absorption filter.

        **Args**:
            - mag_DC (float): The magnitude value of the filter at 0 Hz (linear scale).
            - mag_NY (float): The magnitude value of the filter at Nyquist frequency (linear scale).

        **Returns**:
            - b (torch.Tensor): The numerator coefficients of the filter transfer function.
            - a (torch.Tensor): The denominator coefficients of the filter transfer function.
    """

    b = torch.zeros(2, *(mag_DC.shape))
    a = torch.zeros(2, *(mag_DC.shape))

    r = mag_DC / mag_NY

    a1 = (1 - r) / (1 + r)
    b0 = (1 - a1) * mag_NY

    b[0,:] = b0
    a[0,:] = 1
    a[1,:] = a1

    return b, a


class FDN_one_pole_absorption(dsp.parallelFilter):
    r"""
    Parallel absorption filters for the FDN reverberator.
    """
    def __init__(
        self,
        channels: int=1,
        fs: int = 48000,
        nfft: int = 2**11,
        t60_DC: float = 1.0,
        t60_NY: float = 1.0,
        alias_decay_db: float = 0.0
    ):
        r"""
        Initialize the FDN absorption filters.

            **Args**:
                - channels (int, optional): The number of channels. Defaults to 1.
                - fs (int, optional): The sampling frequency of the signal [Hz]. Defaults to 48000.
                - nfft (int, optional): FFT size. Defaults to 2**11.
                - t60_DC (float, optional): The reverberation time of the FDN at 0 Hz [s]. Defaults to 1.0.
                - t60_NY (float, optional): The reverberation time of the FDN at Nyquist frequency [s]. Defaults to 1.0.
                - alias_decay_db (float, optional): The anti-time-aliasing decay [dB]. Defaults to 0.0.
        """
        super().__init__(size=(1, channels), nfft=nfft, requires_grad=False, alias_decay_db=alias_decay_db)

        self.fs = torch.tensor([fs])
        self.t60_DC = torch.tensor([t60_DC]).repeat(channels)
        self.t60_NY = torch.tensor([t60_NY]).repeat(channels)

    def get_freq_response(self):
        r"""
        Get the frequency response of the absorption filters.
        Reference: flamo.dsp.parallelFilter.get_freq_response()
        """
        self.freq_response = lambda param: self.compute_freq_response(param.squeeze())

    def compute_freq_response(self, delays: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the frequency response of the absorption filters.
        Reference: flamo.dsp.parallelFilter.compute_freq_response()
        """

        absorp_DC = self.rt2absorption(self.t60_DC, self.fs, delays)
        absorp_NY = self.rt2absorption(self.t60_NY, self.fs, delays)

        b, a = one_pole_filter(absorp_DC, absorp_NY)

        b_aa = torch.einsum('p, p... -> p...', (self.gamma ** torch.arange(0, 2, 1)), b)
        a_aa = torch.einsum('p, p... -> p...', (self.gamma ** torch.arange(0, 2, 1)), a)

        B = torch.fft.rfft(b_aa, self.nfft, dim=0)
        A = torch.fft.rfft(a_aa, self.nfft, dim=0)

        return torch.div(B, A)
    
    def rt2absorption(self, rt60: torch.Tensor, fs: int, delay_len: torch.Tensor) -> torch.Tensor:
        r"""
        Convert time in seconds of 60 dB decay to energy decay slope relative to the delay line length.

            **Args**:
                - rt60 (torch.Tensor): The reverberation time [s].
                - fs (int): The sampling frequency of the signal [Hz].
                - delays_len (torch.Tensor): The lengths of the delay lines [samples].

            **Returns**:
                - torch.Tensor: The energy decay slope relative to the delay line length.
        """
        return db2mag(delay_len * rt2slope(rt60, fs))


# def simulate_setup(room_dims: torch.FloatTensor, lds_n: int, mcs_n: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
#     r"""
#     Simulates a RES setup with the given room dimensions and number of loudspeakers and microphones.

#         **Args**:
#             - room_dims (torch.Tensor[float,float,float]): The dimensions of the room [m] - length, width, height.
#             - lds_n (int): The number of loudspeakers.
#             - mcs_n (int): The number of microphones.

#         **Returns**:
#             - torch.Tensor: The loudspeaker positions [m].
#             - torch.Tensor: The microphone positions [m].
#     """
#     assert len(room_dims.shape) == 1, "The room dimensions must be a 1D tensor."
#     assert len(room_dims) == 3, "The room dimensions must have 3 elements."
#     assert torch.all(room_dims > 0), "The room dimensions must be greater than 0."

#     assert isinstance(lds_n, int), "The number of loudspeakers must be an integer."
#     assert isinstance(mcs_n, int), "The number of microphones must be an integer."
#     assert lds_n > 0, "The number of loudspeakers must be greater than 0."
#     assert mcs_n > 0, "The number of microphones must be greater than 0."


#     room_dims = torch.cat([torch.sort(room_dims[0:2], descending=True).values, room_dims[2].unsqueeze(0)], dim=0)
#     surface_n = 5 # shoebox: 4 walls + 1 ceiling
#     # Each surface is defined according to the ranges of values of its x, y, z coordinates, respectively
#     lds_surfaces = [
#         [   [0, room_dims[0]],  [0],                [0, room_dims[2]]   ],
#         [   [room_dims[0], 0],  [room_dims[1]],     [0, room_dims[2]]   ],
#         [   [room_dims[0]],     [0, room_dims[1]],  [0, room_dims[2]]   ],
#         [   [0],                [room_dims[1], 0],  [0, room_dims[2]]   ],
#         [   [0, room_dims[0]],  [0, room_dims[1]],  [room_dims[2]]      ]
#     ]
#     mcs_surfaces = [
#         [   [0],                [0, room_dims[1]],  [room_dims[2], 0]   ],
#         [   [room_dims[0]],     [room_dims[1], 0],  [room_dims[2], 0]   ],
#         [   [0, room_dims[0]],  [room_dims[1]],     [room_dims[2], 0]   ],
#         [   [room_dims[0], 0],  [0],                [room_dims[2], 0]   ],
#         [   [0, room_dims[0]],  [room_dims[1], 0],  [room_dims[2]]      ]
#     ]

#     # Assign an index to each transducer
#     lds_idx = torch.arange(0, lds_n)
#     mcs_idx = torch.arange(0, mcs_n)
#     # Assign each transducer to a surface
#     lds_surface_idx = torch.fmod(lds_idx, surface_n)
#     mcs_surface_idx = torch.fmod(mcs_idx, surface_n)

#     # Allocate the loudspeaker and microphone positions
#     lds_pos = torch.zeros((lds_n, 3))
#     lds_count = 0
#     mcs_pos = torch.zeros((mcs_n, 3))
#     mcs_count = 0

#     # For each surface
#     for i in range(surface_n):
#         # LOUDSPEAKERS
#         # Count the number of loudspeaker in the surface
#         lds_n_on_surface = torch.sum(lds_surface_idx == i)
#         # First iterate along one dimension, then wrap around a second. Third dimension is constant.
#         match i:
#             case 0 | 1:
#                 iter_idx  = 0
#                 wrap_idx  = 2
#                 const_idx = 1
#             case 2 | 3:
#                 iter_idx  = 1
#                 wrap_idx  = 2
#                 const_idx = 0
#             case 4:
#                 iter_idx  = 0
#                 wrap_idx  = 1
#                 const_idx = 2
#             case _:
#                 raise ValueError("Invalid surface index.")
#         # Get the ranges of the coordinates of the surface
#         iterate = lds_surfaces[i][iter_idx ]
#         wrap    = lds_surfaces[i][wrap_idx ]
#         const   = lds_surfaces[i][const_idx]
#         # Allocate the loudspeaker positions
#         lds_pos_on_surface = torch.zeros((lds_n_on_surface, 3))
#         wrap_n = int(torch.ceil(torch.div(lds_n_on_surface, 3))) # Wrap along second dimension in case lds_n_on_surface > 3
#         # Assign the positions of the loudspeakers (max 3 loudspeakers along first dimension)
#         offiter = iterate[1] / (3 * 2)
#         reduciter = iterate[1] / (3)
#         iter_pos = torch.linspace(iterate[0] + offiter, (iterate[1] - reduciter) + offiter, 3).repeat(wrap_n, 1).flatten()
#         offwrap = wrap[1] / ((wrap_n+1) * 2)
#         reducwrap = wrap[1] / (wrap_n+1)
#         wrap_pos = torch.linspace(wrap[0] + offwrap, (wrap[1] - reducwrap) + offwrap, wrap_n).repeat(3, 1).flatten()
#         lds_pos_on_surface[:, iter_idx ] = iter_pos[:lds_n_on_surface]
#         lds_pos_on_surface[:, wrap_idx ] = wrap_pos[:lds_n_on_surface]
#         lds_pos_on_surface[:, const_idx] = const[0]

#         # Assign the positions of the loudspeakers to the global positions
#         lds_pos[lds_count:lds_count+lds_n_on_surface] = lds_pos_on_surface
#         lds_count += lds_n_on_surface

#         # MICROPHONES
#         # Count the number of microphones in the surface
#         mcs_n_on_surface = torch.sum(mcs_surface_idx == i)
#         # First iterate along one dimension, then wrap around a second. Third dimension is constant.
#         match i:
#             case 0 | 1:
#                 iter_idx  = 1
#                 wrap_idx  = 2
#                 const_idx = 0
#             case 2 | 3:
#                 iter_idx  = 0
#                 wrap_idx  = 2
#                 const_idx = 1
#             case 4:
#                 iter_idx  = 0
#                 wrap_idx  = 1
#                 const_idx = 2
#             case _:
#                 raise ValueError("Invalid surface index.")
#         # Get the ranges of the coordinates of the surface
#         iterate = mcs_surfaces[i][iter_idx ]
#         wrap    = mcs_surfaces[i][wrap_idx ]
#         const   = mcs_surfaces[i][const_idx]
#         # Allocate the loudspeaker positions
#         mcs_pos_on_surface = torch.zeros((mcs_n_on_surface, 3))
#         wrap_n = int(torch.ceil(torch.div(mcs_n_on_surface, 3))) # Wrap along second dimension in case mcs_n_on_surface > 3
#         # Assign the positions of the microphones (max 3 microphones along first dimension)
#         offiter = iterate[1] / (2 * 3)
#         reduciter = iterate[1] / (3)
#         iter_pos = torch.linspace(iterate[0] + offiter, (iterate[1] - reduciter) + offiter, 3).repeat(wrap_n, 1).flatten()
#         offwrap = wrap[1] / ((wrap_n+1) * 2)
#         reducwrap = wrap[1] / (wrap_n+1)
#         wrap_pos = torch.linspace(wrap[0] - offiter, (wrap[1] + reducwrap) - offiter, wrap_n).repeat(3, 1).flatten()
#         mcs_pos_on_surface[:, iter_idx ] = iter_pos[:mcs_n_on_surface]
#         mcs_pos_on_surface[:, wrap_idx ] = wrap_pos[:mcs_n_on_surface]
#         mcs_pos_on_surface[:, const_idx] = const[0]

#         # Assign the positions of the microphones to the global positions
#         mcs_pos[mcs_count:mcs_count+mcs_n_on_surface] =  mcs_pos_on_surface
#         mcs_count += mcs_n_on_surface

#     return lds_pos, mcs_pos


def simulate_setup(room_dims: torch.FloatTensor, lds_n: int, mcs_n: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    r"""
    Simulates a RES setup with the given room dimensions and number of loudspeakers and microphones.

        **Args**:
            - room_dims (torch.Tensor[float,float,float]): The dimensions of the room [m] - length, width, height.
            - lds_n (int): The number of loudspeakers.
            - mcs_n (int): The number of microphones.

        **Returns**:
            - torch.Tensor: The loudspeaker positions [m].
            - torch.Tensor: The microphone positions [m].
    """
    assert len(room_dims.shape) == 1, "The room dimensions must be a 1D tensor."
    assert len(room_dims) == 3, "The room dimensions must have 3 elements."
    assert torch.all(room_dims > 0), "The room dimensions must be greater than 0."

    assert isinstance(lds_n, int), "The number of loudspeakers must be an integer."
    assert isinstance(mcs_n, int), "The number of microphones must be an integer."
    assert lds_n > 0, "The number of loudspeakers must be greater than 0."
    assert mcs_n > 0, "The number of microphones must be greater than 0."


    room_dims = torch.cat([torch.sort(room_dims[0:2], descending=True).values, room_dims[2].unsqueeze(0)], dim=0)
    surface_n = 5 # shoebox: 4 walls + 1 ceiling

    # Assign an index to each transducer
    lds_idx = torch.arange(0, lds_n)
    mcs_idx = torch.arange(0, mcs_n)
    # Assign each transducer to a surface
    lds_surface_idx = torch.fmod(lds_idx, surface_n)
    mcs_surface_idx = torch.fmod(mcs_idx, surface_n)

    # Allocate the loudspeaker and microphone positions
    lds_pos = torch.zeros((lds_n, 3))
    lds_count = 0
    mcs_pos = torch.zeros((mcs_n, 3))
    mcs_count = 0

    # For each surface
    for i in range(surface_n):
        # Count the number of loudspeaker and microphones in the surface
        lds_n_on_surface = torch.sum(lds_surface_idx == i)
        mcs_n_on_surface = torch.sum(mcs_surface_idx == i)
        # Case based on the surface index
        match i:
            case 0:
                dim_1_idx = 0
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 1
                const = 0
            case 1:
                dim_1_idx = 0
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 1
                const = room_dims[const_idx]
            case 2:
                dim_1_idx = 1
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 0
                const = 0
            case 3:
                dim_1_idx = 1
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 2
                dim_2 = room_dims[dim_2_idx]
                const_idx = 0
                const = room_dims[const_idx]
            case 4:
                dim_1_idx = 1
                dim_1 = room_dims[dim_1_idx]
                dim_2_idx = 0
                dim_2 = room_dims[dim_1_idx]
                const_idx = 2
                const = room_dims[const_idx]
            case _:
                raise ValueError("Invalid surface index.")
        # Get the positions of loudspeakers and microphones on the current surface
        lds_pos_on_surface, mcs_pos_on_surface = positions_on_surface(dim_1, dim_2, lds_n_on_surface, mcs_n_on_surface)
        # Assign the positions of the loudspeakers to the global positions
        lds_pos[lds_count:lds_count+lds_n_on_surface, dim_1_idx] = lds_pos_on_surface[:, 0]
        lds_pos[lds_count:lds_count+lds_n_on_surface, dim_2_idx] = lds_pos_on_surface[:, 1]
        lds_pos[lds_count:lds_count+lds_n_on_surface, const_idx] = const
        # Assign the positions of the microphones to the global positions
        mcs_pos[mcs_count:mcs_count+mcs_n_on_surface, dim_1_idx] = mcs_pos_on_surface[:, 0]
        mcs_pos[mcs_count:mcs_count+mcs_n_on_surface, dim_2_idx] = mcs_pos_on_surface[:, 1]
        mcs_pos[mcs_count:mcs_count+mcs_n_on_surface, const_idx] = const
        # Update the counts
        lds_count += lds_n_on_surface
        mcs_count += mcs_n_on_surface

    return lds_pos, mcs_pos

def positions_on_surface(dim_1, dim_2, lds_n, mcs_n):
    """
    Evenly distribute microphones and loudspeakers on a surface,
    ensuring they do not share the same position.

    Args:
        dim_1 (float): First dimension of the surface.
        dim_2 (float): Second dimension of the surface.
        mcs_n (int): Number of microphones.
        lds_n (int): Number of loudspeakers.

    Returns:
        mic_positions: (mcs_n, 2) torch.FloatTensor
        speaker_positions: (lds_n, 2) torch.FloatTensor
    """
    total = mcs_n + lds_n
    if total == 0:
        return torch.empty((0, 2)), torch.empty((0, 2))

    # Choose grid resolution
    cols = int(torch.ceil(torch.sqrt(torch.tensor(total)).float()).item())
    rows = int(torch.ceil(total / cols))

    x_vals = torch.linspace(0.5 * dim_1 / cols, dim_1 - 0.5 * dim_1 / cols, cols)
    y_vals = torch.linspace(0.5 * dim_2 / rows, dim_2 - 0.5 * dim_2 / rows, rows)
    yy, xx = torch.meshgrid(y_vals, x_vals, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Compute distances from center of area
    center = torch.tensor([dim_1 / 2, dim_2 / 2])
    dists = torch.norm(grid_points - center, dim=1)

    # Sort grid points by distance to center (closest first)
    sorted_indices = torch.argsort(dists)
    sorted_points = grid_points[sorted_indices][:total]

    # Interleave assignment
    mic_indices = torch.linspace(0, total - 1, mcs_n).round().long()
    all_indices = torch.arange(total)
    speaker_indices = all_indices[~torch.isin(all_indices, mic_indices)]

    mic_positions = sorted_points[mic_indices]
    speaker_positions = sorted_points[speaker_indices]

    return speaker_positions, mic_positions