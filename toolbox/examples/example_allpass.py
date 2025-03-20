import argparse
import os
import time
import matplotlib.pyplot as plt
import torch

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.functional import db2mag, mag2db

from full_system import RES
from physical_room import PhRoom_dataset
from virtual_room import unitary_reverberator
from optimization import system_equalization_curve
from loss_functions import MSE_evs_mod
from plots import plot_evs, plot_spectrograms, plot_raw_evs

torch.manual_seed(130297)


def example_allpass(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*4                  # FFT size
    alias_decay_db = 10                 # Anti-time-aliasing decay in dB

    # Physical room
    rirs_dir = 'LA-lab_1'              # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        dataset_directory="./AA_Dataset/",
        room_name=rirs_dir
    )
    n_stg, n_mcs, n_lds, n_aud = physical_room.get_ems_rcs_number()

    # Virtual room
    t60 = 0.9                       # Unitary allpass reverb reverberation time
    virtual_room = unitary_reverberator(
        n_M = n_mcs,
        n_L = n_lds,
        fs = samplerate,
        nfft = nfft,
        t60 = t60,
        alias_decay_db = alias_decay_db,
    )

    # ------------------- Model Definition --------------------
    model = RES(
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
    
    # ------------------------ Plots --------------------------
    # get VR irs and tfs and plot them

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
    example_allpass(args)