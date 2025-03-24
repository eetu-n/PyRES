import sys
import argparse
import os
import time

import torch

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyRES.full_system import RES
from pyRES.physical_room import PhRoom_dataset
from pyRES.virtual_room import random_FIRs
from pyRES.loss_functions import MSE_evs_mod
from pyRES.utils import system_equalization_curve
from pyRES.plots import plot_evs, plot_spectrograms

torch.manual_seed(130297)


def example_FIRs(args) -> None:

    # -------------------- Initialize RES ---------------------
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
    fir_order = 2**8                   # FIR filter order
    virtual_room = random_FIRs(
        n_M=n_mcs,
        n_L=n_lds,
        fs=samplerate,
        nfft=nfft,
        FIR_order=fir_order,
        alias_decay_db=alias_decay_db,
        requires_grad=True
    )

    # Reverberation Enhancement System
    res = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )

    # ------------------- Model Definition --------------------
    model = system.Shell(
        core=res.open_loop(),
        input_layer=system.Series(
            dsp.FFT(nfft=nfft),
            dsp.Transform(lambda x: x.diag_embed())
        )
    )
    
    # ------------- Performance at initialization -------------
    evs_init = res.open_loop_eigenvalues().squeeze(0)
    ir_init = res.system_simulation().squeeze(0)
    
    # ----------------- Initialize dataset --------------------
    dataset_input = torch.zeros(args.batch_size, samplerate, n_mcs)
    dataset_input[:,0,:] = 1
    dataset_target = system_equalization_curve(evs=evs_init, fs=samplerate, nfft=nfft, f_c=8000)
    dataset_target = dataset_target.view(1,-1,1).expand(args.batch_size, -1, n_mcs)

    dataset = Dataset(
        input = dataset_input,
        target = dataset_target,
        expand = args.num,
        device = args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=args.batch_size, split=args.split, shuffle=False)

    # ------------------- Initialize Trainer ------------------
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device
    )

    # ---------------- Initialize Loss Function ---------------
    criterion = MSE_evs_mod(
        iter_num = args.num,
        freq_points = nfft//2+1,
        samplerate = samplerate,
        lowest_f = 20,
        crossover_freq = 8000,
        highest_f = 15000
    )
    trainer.register_criterion(criterion, 1.0)
    
    # ------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------ Performance after optimization ------------
    evs_opt = res.open_loop_eigenvalues().squeeze(0)
    ir_opt = res.system_simulation().squeeze(0)
    
    # ------------------------ Plots -------------------------
    # TODO: think better about which plots per example
    plot_evs(evs_init, evs_opt, samplerate, nfft, 20, 8000)
    plot_spectrograms(ir_init[:,0], ir_opt[:,0], samplerate, nfft=2**8, noverlap=2**7)

    # ---------------- Save the model parameters -------------
    res.save_state_to(directory='./model_states/')

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
    parser.add_argument('--patience_delta', type=float, default=0.0001, help='Minimum improvement in validation loss to be considered as an improvement')
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
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
    example_FIRs(args)