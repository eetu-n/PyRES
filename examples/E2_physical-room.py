# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyTorch
import torch
# PyRES
from PyRES.physical_room import PhRoom_dataset, PhRoom_wgn


###########################################################################################
# In this example we implement a physical room by either loading it from a dataset or simulating it.
# The PhRoom class is a base class used to represent a physical room in PyRES. Its subclasses
# PhRoom_dataset and PhRoom_wgn implement physical rooms by loading data from a dataset or
# simulating the room using exponentially decaying white Gaussian noise, respectively.
# This class requires the following parameters:
# - fs: sampling frequency [Hz]
# - nfft: FFT size
# - alias_decay_db: anti-time-aliasing decay [Hz]
# - dataset_directory: path to the dataset
# - room_name: path to the room impulse responses
# The PhRoom_dataset class implements a physical room by interfacing with dataRES.
# Reference:
#   De Bortoli, G. M. (2025).
#   DataRES: Dataset for research on Reverberation Enhancement Systems (1.0.0) [Data set]. Zenodo.
#   https://doi.org/10.5281/zenodo.15165524
###########################################################################################

torch.manual_seed(12345)

def room_from_dataset(
        fs: int,
        nfft: int,
        alias_decay_db: int,
        dataset_directory: str = './dataRES',
        room_name: str = 'Otala'
    ):
    """
    Loads a physical room from a dataset.
    
    Parameters:
        fs (int): Sampling frequency [Hz].
        nfft (int): FFT size.
        alias_decay_db (int): Anti-time-aliasing decay [dB].
        dataset_directory (str): Path to the dataset.
        room_name (str): Name of the room in the dataset.
    
    Returns:
        PhRoom_dataset: Instance of the physical room.
    """
    physical_room = PhRoom_dataset(
        fs=fs,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )

    return physical_room

def simulate_room(
        fs: int,
        nfft: int,
        alias_decay_db: int,
        room_dims: tuple[float, float, float] = (12.1, 8.5, 3.2),
        room_RT: float = 0.7,
        n_L: int = 16,
        n_M: int = 8
    ):
    """
    Simulates a physical room by approximating the RIRs with decaying white Gaussian noise.
    
    Parameters:
        fs (int): Sampling frequency [Hz].
        nfft (int): FFT size.
        alias_decay_db (int): Anti-time-aliasing decay [dB].
        room_dims (tuple[float, float, float]): Dimensions of the room (length, width, height) in meters.
        room_RT (float): Reverberation time of the room [seconds].
        n_L (int): Number of loudspeakers.
        n_M (int): Number of microphones.
    
    Returns:
        PhRoom_dataset: Instance of the physical room.
    """
    physical_room = PhRoom_wgn(
        fs=fs,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        room_dims=room_dims,
        room_RT=room_RT,
        n_L=n_L,
        n_M=n_M
    )

    return physical_room

if __name__ == '__main__':

    # Time-frequency
    samplerate = 48000              # Sampling frequency
    nfft = samplerate*3             # FFT size
    alias_decay_db = 0              # Anti-time-aliasing decay in dB

    # Physical room
    physical_space
    

    physical_room.plot_setup()
    physical_room.plot_coupling()
    physical_room.plot_DRR()