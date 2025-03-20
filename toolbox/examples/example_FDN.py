import argparse
import os
import time
import matplotlib.pyplot as plt
import torch

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.functional import db2mag, mag2db

from full_system import AAES
from physical_room import PhRoom_dataset
from virtual_room import feedback_delay_network
from optimization import system_equalization_curve
from loss_functions import MSE_evs_mod
from plots import plot_evs, plot_spectrograms, plot_raw_evs

torch.manual_seed(130297)


def example_FDN(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*2                  # FFT size
    alias_decay_db = -10                 # Anti-time-aliasing decay in dB

    # Physical room
    rirs_dir = 'LA-lab_1'              # Path to the room impulse responses
    physical_room = PhRoom_dataset(fs=samplerate, room_name=rirs_dir)
    srs_rcs = physical_room.get_ems_rcs_number()
    n_stg = srs_rcs['n_S']             # Number of sources
    n_mcs = srs_rcs['n_M']             # Number of microphones
    n_lds = srs_rcs['n_L']             # Number of loudspeakers
    n_aud = srs_rcs['n_A']             # Number of audience positions

    # Virtual room
    t60_DC = 1.5                       # FDN reverberation time at 0 Hz
    t60_Ny = 0.5                       # FDN reverberation time at Nyquist frequency
    virtual_room = feedback_delay_network(
        n_M = n_mcs,
        n_L = n_lds,
        fs = samplerate,
        nfft = nfft,
        t60_DC = t60_DC,
        t60_NY = t60_Ny,
        alias_decay_db = alias_decay_db,
        requires_grad = True
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
        alias_decay_db = alias_decay_db
    )

    # Apply safe margin
    gbi_init = model.compute_GBI()
    model.set_G(db2mag(mag2db(gbi_init) - 2))
    
    # ------------- Performance at initialization -------------
    # Performance metrics
    evs_init = model.open_loop_eigenvalues().squeeze(0)
    ir_init = model.system_simulation().squeeze(0)
    
    # ------------------------ Plots -------------------------
    # Get VR irs and tfs and plot them

    # --------------------- Save state ------------------------
    torch.save(model.get_state(), os.path.join('./toolbox/optimization/', time.strftime("%Y-%m-%d_%H.%M.%S.pt")))
    
    return None


###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**4,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=0.0001, help='Minimum improvement in validation loss to be considered as an improvement')
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
    example_FDN(args)