# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyTorch
import torch
# PyRES
from PyRES.virtual_room import unitary_mixing_matrix


###########################################################################################
# In this example we implement a virtual room using the unitary_mixing_matrix class.
# This class requires the following parameters:
# - n_M: "number of microphones" in the physical room. Since there is no physical room
#        in this example, we will call this parameter "n_inputs"
# - n_L: "number of loudspeakers" in the physical room. Since there is no physical room
#        in this example, we will call this parameter "n_outputs"
# - fs: sampling frequency [Hz]
# - nfft: FFT size
# - alias_decay_db: anti-time-aliasing decay [Hz]
# The unitary_mixing_matrix class implements a mixing matrix that has arbitrary dimensions
# and is "pseudo-unitary" in the sense that it is not guaranteed to be unitary.
# The matrix is unitary if inputs==outputs, otherwise is a truncated unitary matrix.
# The unitary_mixing_matrix class is can be used in RES for equalization or reverberation
# gain increase. The reverberation time increase is dependent on the system gain.
###########################################################################################

torch.manual_seed(12345)

if __name__ == '__main__':

    # Parameters
    samplerate = 48000           # Sampling frequency
    nfft = samplerate*3          # FFT size
    alias_decay_db = 0           # Anti-time-aliasing decay
    n_inputs = 4                   # Number of inputs
    n_outputs = 4                  # Number of outputs
    
    # Virtual room
    virtual_room = unitary_mixing_matrix(
        n_M = n_inputs,
        n_L = n_outputs,
        fs = samplerate,
        nfft = nfft,
        alias_decay_db = alias_decay_db,
    )