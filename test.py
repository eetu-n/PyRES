import sys
import os
import time 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import torch.nn as nn

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.optimize.loss import mss_loss

from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.plots import plot_irs_compare

torch.manual_seed(141122)

if __name__ == '__main__':
    samplerate = 48000
    nfft = samplerate
    alias_decay_db = 0.0
    FIR_order = 2**12
    lr = 1e-3 
    epochs = 20

    dataset_directory = './dataRES'
    room_name = 'MarsioExperimentalStudio3MicSetup2'
    train_dir = 'training_output'

    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    n_M = physical_room.transducer_number['mcs']  # Number of microphones
    n_L = physical_room.transducer_number['lds']  # Number of loudspeakers

    print(f"Number of Mics:", n_M)
    print(f"Number of Loudspeakers:", n_L)

    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    virtual_room = random_FIRs(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=FIR_order,
        requires_grad=False
    )

    res = RES(physical_room=physical_room, virtual_room=virtual_room)

    states = res.get_v_ML_state()
    res.set_v_ML_state(states)

    input_signal = torch.zeros(1, samplerate, 1)
    input_signal[:,0,:] = 1

    