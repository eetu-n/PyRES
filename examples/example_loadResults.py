import argparse
import os
import time
import matplotlib.pyplot as plt
import torch

from flamo.functional import db2mag, mag2db

from full_system import AAES
from physical_room import PhRoom_dataset
from virtual_room import random_FIRs, phase_canceling_modal_reverb
from plots import plot_coupling, plot_DRR, plot_evs, plot_spectrograms

torch.manual_seed(130297)


def previous_opt(args) -> None:
    r"""
    Active Acoustics training test function.
    Training results are plotted showing the difference in performance between the initialized model and the optimized model.
    The model parameters are saved to file.
    You can modify the number of microphones (should be set between 1 and 4) and the number of loudspeakers (should be set between 1 and 13).
    Please use n_S = 1 and  n_A = 1.
    Measured room impulse responses for additional source and/or audience positions are not available.

        **Args**:
            A dictionary or object containing the necessary arguments for the function.
    """

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*2                  # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    # Physical room
    rirs_dir = 'LA-lab_1'              # Path to the room impulse responses
    physical_room = PhRoom_dataset(fs=samplerate, room_name=rirs_dir)
    srs_rcs = physical_room.get_ems_rcs_number()
    n_stg = srs_rcs['n_S']             # Number of sources
    n_mcs = srs_rcs['n_M']             # Number of microphones
    n_lds = srs_rcs['n_L']             # Number of loudspeakers
    n_aud = srs_rcs['n_A']             # Number of audience positions

    # Virtual room
    fir_order = 2**8                   # FIR filter order
    virtual_room = random_FIRs(
        n_M=n_mcs,
        n_L=n_lds,
        nfft=nfft,
        FIR_order=fir_order,
        alias_decay_db=alias_decay_db,
        requires_grad=True
    )

    # ------------------- Model Definition --------------------
    model = AAES(
        n_S = n_stg,
        n_M = n_mcs,
        n_L = n_lds,
        n_A = n_aud,
        fs = samplerate,
        nfft = nfft,
        physical_room = physical_room,
        virtual_room = virtual_room,
        alias_decay_db=alias_decay_db
    )
    
    # Apply safe margin
    gbi_init = model.compute_GBI()
    model.set_G(db2mag(mag2db(gbi_init) - 2))
    
    # ------------- Performance at initialization -------------

    # Performance metrics
    evs_init = model.open_loop_eigenvalues().squeeze(0)
    ir_init = model.system_simulation().squeeze(0)

    # -------------=--- Load optimized state ------------------
    state_opt = torch.load(os.path.join('./toolbox/optimization/FIRs.pt'))
    model.set_state(state_opt)

    # ------------ Performance after optimization -------------

    # Performance metrics
    evs_opt = model.open_loop_eigenvalues().squeeze(0)
    ir_opt = model.system_simulation().squeeze(0)
    
    # ------------------------ Plots --------------------------
    plot_coupling(physical_room.get_lds_to_mcs())
    plot_DRR(physical_room.get_lds_to_mcs(), fs=samplerate)
    plot_evs(evs_init, evs_opt, samplerate, nfft, 20, 9000)
    plot_spectrograms(ir_init[:,0], ir_opt[:,0], samplerate, nfft=2**8, noverlap=2**7)
    
    return None

###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**8,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=0.005, help='Minimum improvement in validation loss to be considered as an improvement')
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    #----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments 
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Run examples
    previous_opt(args)