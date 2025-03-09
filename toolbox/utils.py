

def save_model_params(model: nn.Module, filename: str='parameters'):
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


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()