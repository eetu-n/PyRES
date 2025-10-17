import argparse
import time
import sys
import os
# PyTorch
import torch
# FLAMO
from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
# PyRES
from PyRES import virtual_room
from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.loss_functions import MSE_evs_idxs, colorless_reverb
from PyRES.plots import plot_evs_compare, plot_spectrograms_compare

def train_deverb(args) -> None:
    fs = 1000
    nfft = fs*3
    alias_decay_db = -20

    physical_room = PhRoom_dataset(
        fs=fs,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory= './dataRES',
        room_name = 'Otala'
    )

    num_mic = physical_room.transducer_number['mcs']  # Number of microphones
    num_ls = physical_room.transducer_number['lds']  # Number of loudspeakers

    virtual_room = random_FIRs(
        n_M = num_mic,
        n_L = num_ls,
        fs = fs,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
        FIR_order = 10,
        requires_grad = True
    )
    
    res = RES(physical_room, virtual_room)

    model = system.Shell(
        core = res.open_loop(),
        input_layer = system.Series(
            dsp.FFT(nfft),
            dsp.Transform(lambda x: x.diag_embed())
        )
    )

    evs_init = res.open_loop_eigenvalues()
    _,_,ir_init = res.system_simulation()


    dataset_input = torch.zeros(1, nfft//2+1, num_mic)
    dataset_input[:,0,:] = 1
    dataset_target = torch.zeros(1, nfft//2+1, num_mic)
    dataset = Dataset(
        input = dataset_input,
        target = dataset_target,
        expand = args.num,
        device = args.device
        )
    train_loader, valid_loader  = load_dataset(dataset, batch_size=1, split=args.split, shuffle=False)

    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device
    )

    MR_freqs = virtual_room.get_v_ML()[0].resonances[:,0,0].clone().detach()
    criterion1 = MSE_evs_idxs(
        iter_num = args.num,
        freq_points = (nfft // 2) + 1,
        samplerate = fs,
        freqs = MR_freqs
    )

    trainer.register_criterion(criterion1)

    trainer.train(train_loader, valid_loader)

    