# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import numpy as np
import pyfar as pf
# Torch
import torch


def reverb_time(rirs: torch.Tensor) -> torch.Tensor:

    pf_rirs = pf.Signal(rirs.numpy(), rirs.shape[-1], 1)
    rt = pf.reverberation_time(pf_rirs, method='t30')

    return torch.tensor(rt)

def peak_to_mean_ratio(array: torch.Tensor, dim: tuple[int]=None) -> torch.Tensor:
    f"""
    Computes the peak to mean ratio of an array along a given dimension.

        **Args**:
            array (torch.Tensor): Input array.
            dim (int, tuple): Dimension(s) along which to compute the peak to mean ratio.

        **Returns**:
            torch.Tensor: Peak to mean ratio.
    """

    if dim is None:
        dim = tuple(range(1, array.ndim))
    else:
        if isinstance(dim, int):
            dim = (dim,)
        elif not isinstance(dim, tuple):
            raise ValueError("dim must be an int or a tuple of ints")

    max_vals = torch.amax(input=array, dim=dim, keepdim=True)
    mean_vals = torch.mean(input=array, dim=dim, keepdim=True)

    return max_vals / mean_vals


def coloration_coefficient(ir: torch.Tensor, interval: tuple[int,int]=None) -> torch.Tensor:
    f"""
    Computes the coloration coefficient of an array.

        **Args**:
            ir (torch.Tensor): impulse response.

        **Returns**:
            torch.Tensor: Coloration coefficient as the standard deviation with respect to a mean of 1.
    """
    if interval is not None:
        array = array[interval[0]:interval[1]]

    mean_value = torch.mean(input=array, dim=0, keepdim=True)
    array_norm = array / mean_value

    return torch.std(input=array_norm, dim=0, keepdim=True)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    array = torch.randn(100, 10, 10)

    ptmr = peak_to_mean_ratio(array)

    cc = coloration_coefficient(array)

    cc1 = coloration_coefficient(array[:,0,0].squeeze())

    a = 1