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
from virtual_room import phase_canceling_modal_reverb
from loss_functions import MSE_evs_idxs, colorless_reverb
from plots import plot_raw_evs, plot_evs, plot_spectrograms

torch.manual_seed(130297)


def example_phase_cancellation(args) -> None:

    # --------------------- Parameters ------------------------
    # Time-frequency
    samplerate = 1000                 # Sampling frequency
    nfft = samplerate*2                  # FFT size
    alias_decay_db = -20                 # Anti-time-aliasing decay in dB

    # Physical room
    rirs_dir = 'LA-lab_1'              # Path to the room impulse responses
    physical_room = PhRoom_dataset(fs=samplerate, room_name=rirs_dir)
    srs_rcs = physical_room.get_ems_rcs_number()
    n_stg = srs_rcs['n_S']             # Number of sources
    n_mcs = srs_rcs['n_M']             # Number of microphones
    n_lds = srs_rcs['n_L']             # Number of loudspeakers
    n_aud = srs_rcs['n_A']             # Number of audience positions

    # Virtual room
    MR_n_modes = 150                   # Modal reverb number of modes
    MR_f_low = 50                      # Modal reverb lowest mode frequency
    MR_f_high = 450                    # Modal reverb highest mode frequency
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
        alias_decay_db = alias_decay_db
    )

    # Apply safe margin
    gbi_init = model.compute_GBI()
    model.set_G(db2mag(mag2db(gbi_init) - 2))
    
    # ------------- Performance at initialization -------------
    # Performance metrics
    evs_init = model.open_loop_eigenvalues().squeeze(0)
    ir_init = model.system_simulation().squeeze(0)

    # ---------------- Define optimization --------------------
    model.set_forward_inputLayer(system.Series(
        dsp.Transform(lambda x: x.diag_embed()),
        dsp.FFT(nfft)
        ))

    # ----------------- Initialize dataset --------------------
    dataset_input = torch.zeros(args.batch_size, nfft//2+1, n_mcs)
    dataset_input[:,0,:] = 1
    dataset_target = torch.zeros(args.batch_size, nfft//2+1, n_mcs)
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
    MR_freqs = virtual_room.resonances[:,0,0].clone().detach()
    criterion1 = MSE_evs_idxs(
        iter_num = args.num,
        freq_points = nfft//2+1,
        samplerate = samplerate,
        freqs = MR_freqs
    )
    trainer.register_criterion(criterion1, 1.5)
    criterion2 = colorless_reverb(
        samplerate = samplerate,
        freq_points = nfft//2+1,
        freqs = MR_freqs
    )
    trainer.register_criterion(criterion2, 0.5, requires_model=True)
    
    # -------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------- Performance after optimization ------------
    # Save the model state
    # save_model_params(model, filename='AA_parameters_optim')

    # Performance metrics
    evs_opt = model.open_loop_eigenvalues().squeeze(0)
    ir_opt = model.system_simulation().squeeze(0)
    
    # ------------------------- Plots -------------------------
    plot_raw_evs(evs_init, evs_opt)
    plot_evs(evs_init, evs_opt, samplerate, nfft, 20, 480)
    plot_spectrograms(ir_init[:,0], ir_opt[:,0], samplerate, nfft=2**4, noverlap=2**3)

    torch.save(model.get_state(), os.path.join('./toolbox/optimization/', time.strftime("%Y-%m-%d_%H.%M.%S.pt")))

    return None


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
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
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
    example_phase_cancellation(args)