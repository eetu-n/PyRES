import argparse
import os
import time
import torch

from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer

from .full_system import AAES
from .virtual_room import random_FIRs, phase_canceling_modal_reverb
from .loss_functions import minimize_evs_mod, preserve_reverb_energy
from .plots import plot_evs_distributions, plot_spectrograms
from .utils import next_power_of_2, save_model_params

torch.manual_seed(130297)


def example_measured_room(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 1000                  # Sampling frequency
    nfft = samplerate*2                # FFT size

    # Physical room
    n_src = 0                          # Number of stage sources
    n_mcs = 16                         # Number of microphones
    n_lds = 16                         # Number of loudspeakers
    n_aud = 0                          # Number of audience receivers
    rirs_dir = './rirs/LA-lab'         # Path to the room impulse responses

    # Virtual room
    MR_n_modes = 160                   # Modal reverb number of modes
    MR_f_low = 40                      # Modal reverb lowest mode frequency
    MR_f_high = 480                    # Modal reverb highest mode frequency
    MR_t60 = 0.5                       # Modal reverb reverberation time
    virtual_room = phase_canceling_modal_reverb(
        size=(n_mcs, n_lds),
        nfft=nfft,
        fs=samplerate,
        n_modes=MR_n_modes,
        low_f_lim=MR_f_low,
        high_f_lim=MR_f_high,
        t60=MR_t60,
        requires_grad=True
    )

    # Loss function
    f_axis = torch.linspace(0, samplerate/2, nfft//2+1)
    MR_freqs = torch.linspace(MR_f_low, MR_f_high, MR_n_modes)
    idxs = torch.argmin(torch.abs(f_axis.unsqueeze(1) - MR_freqs.unsqueeze(0)), dim=0)
    idxs = torch.cat((idxs-2, idxs-1, idxs, idxs+1, idxs+2), dim = 0)

    # ------------------- Model Definition --------------------
    model = AAES(
        n_S = n_src,
        n_M = n_mcs,
        n_L = n_lds,
        n_A = n_aud,
        fs = samplerate,
        nfft = nfft,
        room_name = rirs_dir,
        V_ML = virtual_room,
        alias_decay_db=-30
    )

    # Save the model state
    save_model_params(model, filename='AA_parameters_init')
    
    # ------------- Performance at initialization -------------
    # Performance metrics
    evs_init = model.get_open_loop_eigenvalues().squeeze(0)
    ir_init = model.system_simulation().squeeze(0)

    # ----------------- Initialize dataset --------------------
    dataset = DatasetColorless(
        input_shape = (args.batch_size, nfft//2+1, n_mcs),
        target_shape = (args.batch_size, nfft//2+1, n_lds, n_mcs),
        expand = args.num,
        device = args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=args.batch_size, split=args.split, shuffle=False)

    # ------------- Initialize training process ---------------
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device
    )
    trainer.register_criterion(minimize_evs_mod(iter_num=args.num, idxs=idxs), 1.5)
    trainer.register_criterion(preserve_reverb_energy(idxs), 0.5, requires_model=True)
    
    # ------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # Save the model state
    save_model_params(model, filename='AA_parameters_optim')

    # ------------ Performance after optimization ------------
    # Performance metrics
    evs_opt = model.get_open_loop_eigenvalues().squeeze(0)
    ir_opt = model.system_simulation().squeeze(0)
    
    # ------------------------ Plots -------------------------
    plot_evs_distributions(evs_init, evs_opt, samplerate, nfft, MR_f_low, MR_f_high)
    plot_spectrograms(ir_init, ir_opt, samplerate, nfft=2**4)

    return None


def example_ideal_room(args):
    pass


###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**6,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.8, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=0.01, help='Minimum improvement in validation loss to be considered as an improvement')
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
    example(args)