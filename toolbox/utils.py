# ==================================================================
# ============================ IMPORTS =============================
# Torch
import torch


def next_power_of_2(x): 
    r"""
    Returns the next power of 2 of the input number.

        **Args**:
            x (int): Input number.

        **Returns**:
            int: Next power of 2.
    """
    return 1 if x == 0 else 2**(x - 1).bit_length()


def limit_frequency_points(array: torch.Tensor, fs: int, nfft: int, f_interval: tuple[float, float]=None, f_subset: torch.Tensor=None) -> torch.Tensor:
    f"""
    Reduces the input array to a given frequency interval or to a given frequency subset.

        **Args**:
            array (torch.Tensor): Input array.
            fs (int): Sampling frequency [Hz].
            nfft (int): Number of frequency bins.
            f_interval (tuple[float, float], optional): Frequency interval [Hz]. Defaults to None.
            f_subset (torch.Tensor, optional): Frequency points [Hz]. Defaults to None.

        **Returns**:
            torch.Tensor: reduced array.
    """
    
    if f_interval is not None:
        freqs = torch.linspace(0, fs/2, nfft//2+1)
        index_1 = torch.argmin(torch.abs(freqs - torch.tensor(f_interval[0])))
        index_2 = torch.argmin(torch.abs(freqs - torch.tensor(f_interval[1])))
        return array[index_1:index_2+1]
    elif f_subset is not None:
        freqs = torch.linspace(0, fs/2, nfft//2+1)
        index_subset = torch.argmin(torch.abs(freqs - f_subset.unsqueeze(0)), dim=1)
        return array[index_subset]
    else:
        return array