import sys
import argparse
import os
import time

import torch

from flamo import system

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyRES.full_system import RES
from pyRES.physical_room import PhRoom_dataset
from pyRES.virtual_room import unitary_reverberator
from pyRES.plots import plot_virtualroom_ir

torch.manual_seed(130297)


def example_unitary_reverb(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*3                # FFT size
    alias_decay_db = 0                 # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './AA_dataset'      # Path to the dataset
    room = 'LA-lab_1'                  # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    _, n_mcs, n_lds, _ = physical_room.get_ems_rcs_number()

    # Virtual room
    t60 = 1.0                          # Unitary allpass reverb reverberation time
    virtual_room = unitary_reverberator(
        n_M = n_mcs,
        n_L = n_lds,
        fs = samplerate,
        nfft = nfft,
        t60 = t60,
        alias_decay_db = alias_decay_db,
    )

    # Reverberation Enhancement System
    res = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )

    # ------------- Performance at initialization -------------
    # evs_init = res.open_loop_eigenvalues()
    # ir_init = res.system_simulation()

    # ------------ Performance after optimization ------------
    # evs_opt = res.open_loop_eigenvalues()
    # ir_opt = res.system_simulation()
    
    # ------------------------ Plots -------------------------
    test = system.Shell(core=virtual_room)
    irs = test.get_time_response(identity=True).squeeze()
    plot_virtualroom_ir(ir=irs[:,0,0], fs=samplerate, nfft=nfft)
    plot_virtualroom_ir(ir=irs[:,2,1], fs=samplerate, nfft=nfft)
    plot_virtualroom_ir(ir=irs[:,1,3], fs=samplerate, nfft=nfft)

    # ---------------- Save the model parameters -------------
    res.save_state_to(directory='./model_states/')

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
    example_unitary_reverb(args)