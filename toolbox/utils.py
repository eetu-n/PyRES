import os
import scipy.io
import torch
import torch.nn as nn


def next_power_of_2(x):  
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
    # TODO: generalize to input array of any shape
    if f_interval is not None:
        freqs = torch.linspace(0, fs/2, nfft//2+1)
        index_interval = torch.argmin(torch.abs(freqs - torch.tensor(f_interval)))
        return array[index_interval[0]:index_interval[1]+1,:,:]
    elif f_subset is not None:
        freqs = torch.linspace(0, fs/2, nfft//2+1)
        index_subset = torch.argmin(torch.abs(freqs - f_subset.unsqueeze(0)), dim=1)
        return array[index_subset,:,:]
    else:
        return array


def save_model_params(model: nn.Module, args: dict, filename: str='parameters'):
    r"""
    Retrieves the parameters of a feedback delay network (FDN) from a given network and saves them in .mat format.

        **Parameters**:
            model (Shell): The Shell class containing the FDN.
            filename (str): The name of the file to save the parameters without file extension. Defaults to 'parameters'.
        **Returns**:
            dict: A dictionary containing the FDN parameters.
                - 'FIR_matrix' (ndarray): The FIR matrix.
                - 'WGN_reverb' (ndarray): The WGN reverb.
                - 'G' (ndarray): The general gain.
                - 'H_LM' (ndarray): The loudspeakers to microphones RIRs.
                - 'H_LA' (ndarray): The loudspeakers to audience RIRs.
                - 'H_SM' (ndarray): The sources to microphones RIRs.
                - 'H_SA' (ndarray): The sources to audience RIRs.
    """

    param = {}
    param['FIR_matrix'] = model.V_ML['U'].param.squeeze().detach().clone().numpy()
    param['WGN_reverb'] = model.V_ML['R'].param.squeeze().detach().clone().numpy()
    param['G'] = model.G.param.squeeze().detach().clone().numpy()
    param['H_LM'] = model.H_LM.param.squeeze().detach().clone().numpy()
    param['H_LA'] = model.H_LA.param.squeeze().detach().clone().numpy()
    param['H_SM'] = model.H_SM.param.squeeze().detach().clone().numpy()
    param['H_SA'] = model.H_SA.param.squeeze().detach().clone().numpy()
    
    scipy.io.savemat(os.path.join(args.train_dir, filename + '.mat'), param)

    return param
