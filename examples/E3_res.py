# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyTorch
import torch
# PyRES
from PyRES.virtual_room import unitary_reverberator
from PyRES.physical_room import PhRoom_dataset
from PyRES.res import RES


###########################################################################################
# In this example we implement a Reverberation Enhancement System (RES) using the RES class.
# This class requires the following parameters:
# - physical_room: physical room object
# - virtual_room: virtual room object
# The RES class possess several methods to control the system (see RES class documentation).
###########################################################################################

torch.manual_seed(12345)

if __name__ == '__main__':

    # Time-frequency
    samplerate = 48000               # Sampling frequency
    nfft = samplerate*3              # FFT size
    alias_decay_db = 0               # Anti-time-aliasing decay in dB

    # Physical room
    room_dataset = './dataRES'       # Path to the dataset
    room = 'Otala'                   # Path to the room impulse responses
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=room_dataset,
        room_name=room
    )
    _, n_M, n_L, _ = physical_room.get_ems_rcs_number()
    
    # Virtual room
    t60 = 1.5                        # Reverberation time at 0 Hz
    virtual_room = unitary_reverberator(
        n_M = n_M,
        n_L = n_L,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
        t60 = t60,
    )

    # Reverberation Enhancement System
    res = RES(
        physical_room = physical_room,
        virtual_room = virtual_room
    )