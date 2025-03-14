import argparse
import os
import time
import torch

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.functional import db2mag, mag2db, get_eigenvalues, get_magnitude

from full_system import AAES
from physical_room import PhRoom_dataset, PhRoom_ideal
from virtual_room import random_FIRs, phase_canceling_modal_reverb
from optimization import system_equalization_curve
from loss_functions import evs_mse_loss, masked_mse_loss, colorless_reverb
from plots import plot_evs_distributions, plot_spectrograms, plot_coupling
from utils import next_power_of_2, limit_frequency_points, save_model_params

torch.manual_seed(130297)


def example_phase_cancellation(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 1000                  # Sampling frequency
    nfft = samplerate*2                # FFT size

    # Physical room
    rirs_dir = 'LA-lab_2024.11.12_noEQ'         # Path to the room impulse responses
    physical_room = PhRoom_dataset(fs=samplerate, room_name=rirs_dir)
    srs_rcs = physical_room.srs_rcs_number()
    n_stg = srs_rcs[0]                 # Number of sources
    n_mcs = srs_rcs[1]                 # Number of microphones
    n_lds = srs_rcs[2]                 # Number of loudspeakers
    n_aud = srs_rcs[3]                 # Number of audience positions

    # Virtual room
    MR_n_modes = 160                   # Modal reverb number of modes
    MR_f_low = 40                      # Modal reverb lowest mode frequency
    MR_f_high = 480                    # Modal reverb highest mode frequency
    MR_t60 = 1.00                      # Modal reverb reverberation time
    virtual_room = phase_canceling_modal_reverb(
        n_M=n_mcs,
        n_L=n_lds,
        fs=samplerate,
        nfft=nfft,
        n_modes=MR_n_modes,
        low_f_lim=MR_f_low,
        high_f_lim=MR_f_high,
        t60=MR_t60,
        requires_grad=True,
        alias_decay_db=-20
    )

    # ------------------- Model Definition --------------------
    model = AAES(
        n_S = n_stg,
        n_M = n_mcs,
        n_L = n_lds,
        n_A = n_aud,
        fs = samplerate,
        nfft = nfft,
        room_name = physical_room,
        V_ML = virtual_room,
        alias_decay_db=-30
    )
    
    # ------------- Performance at initialization -------------
    # Save the model state
    # save_model_params(model, filename='AA_parameters_init')

    # Performance metrics
    evs_init = model.open_loop_eigenvalues().squeeze(0)
    ir_init = model.system_simulation().squeeze(0)

    # ----------------- Initialize dataset --------------------
    dataset_input = torch.zeros(args.batch_size, nfft//2+1, n_mcs)
    dataset_input[:,0,:] = 1
    dataset_target = torch.zeros(args.batch_size, nfft//2+1, n_lds, n_mcs)
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
    f_axis = torch.linspace(0, samplerate/2, nfft//2+1)
    MR_freqs = torch.linspace(MR_f_low, MR_f_high, MR_n_modes)
    idxs = torch.argmin(torch.abs(f_axis.unsqueeze(1) - MR_freqs.unsqueeze(0)), dim=0)
    idxs = torch.cat((idxs-2, idxs-1, idxs, idxs+1, idxs+2), dim = 0)

    trainer.register_criterion(evs_mse(iter_num=args.num, freq_points=nfft), 1.5)
    trainer.register_criterion(colorless_reverb(idxs), 0.5, requires_model=True)
    
    # -------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------- Performance after optimization ------------
    # Save the model state
    # save_model_params(model, filename='AA_parameters_optim')

    # Performance metrics
    evs_opt = model.get_open_loop_eigenvalues().squeeze(0)
    ir_opt = model.system_simulation().squeeze(0)
    
    # ------------------------- Plots -------------------------
    plot_evs_distributions(evs_init, evs_opt, samplerate, nfft, MR_f_low, MR_f_high)
    plot_spectrograms(ir_init, ir_opt, samplerate, nfft=2**4)

    return None


def example_FIRs(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 48000                 # Sampling frequency
    nfft = samplerate*2                # FFT size
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
    fir_order = 100                    # FIR filter order
    virtual_room = random_FIRs(
        n_M=n_mcs,
        n_L=n_lds,
        nfft=nfft,
        FIR_order=fir_order,
        alias_decay_db=alias_decay_db
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
    # Save the model state
    # save_model_params(model, filename='AA_parameters_init')

    # Performance metrics
    evs_init = model.open_loop_eigenvalues().squeeze(0)
    ir_init = model.system_simulation().squeeze(0)

    # ---------------- Define optimization --------------------
    model.set_forward_inputLayer(system.Series(
        dsp.Transform(lambda x: x.diag_embed()),
        dsp.FFT(nfft)
        ))
    model.set_forward_outputLayer(system.Series(
        dsp.Transform(lambda x: get_eigenvalues(x)),
        dsp.Transform(lambda x: get_magnitude(x)),
        dsp.Transform(lambda x: limit_frequency_points(x, samplerate, nfft, f_interval=(20, 20000)))
        ))
    
    # ----------------- Initialize dataset --------------------
    dataset_input = torch.zeros(args.batch_size, nfft//2+1, n_mcs)
    dataset_input[:,0,:] = 1
    dataset_target = system_equalization_curve(evs_init, samplerate, nfft, f_crossover = 8000)
    dataset_target = limit_frequency_points(dataset_target, samplerate, nfft, f_interval=(20, 20000))
    dataset_target.unsqueeze(0).unsqueeze(2).expand(args.batch_size, -1, n_mcs)

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
    criterion = masked_mse_loss(
        nfft=nfft,
        samples_in_partition=2**10,
        device=args.device
    )
    trainer.register_criterion(criterion, 1.0)
    
    # ------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # Apply safe margin
    gbi_opt = model.compute_GBI()
    model.set_G(db2mag(mag2db(gbi_opt) - 2))

    # ------------ Performance after optimization ------------
    # Save the model state
    # save_model_params(model, filename='AA_parameters_optim')

    # Performance metrics
    evs_opt = model.open_loop_eigenvalues().squeeze(0)
    ir_opt = model.system_simulation().squeeze(0)
    
    # ------------------------ Plots -------------------------
    plot_evs_distributions(evs_init, evs_opt, samplerate, nfft, 20, 20000)
    plot_spectrograms(ir_init, ir_opt, samplerate, nfft=2**8, noverlap=2**7)

    return None


def example_ideal_room(args):
    pass


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
    example_FIRs(args)