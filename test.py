import sys
import os
import numpy as np
import scipy.io.wavfile as wavfile 
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
import torch.nn as nn

from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo_trainer import Trainer
 
import flamo.optimize.loss as loss

from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.plots import plot_irs_compare, plot_spectrograms_compare
from PyRES.loss_functions import BruteForceDirectPath, PunishHighValues

from loss_funcs import ESRLoss
#torch.manual_seed(141122)

def convolve(signal_path, ir):

    data, sr = wavfile.read(signal_path)

    y = np.convolve(data, ir, mode='full')
    return y


def plot_spectrograms(signal1, signal2, sr):
    """
    Plots two spectrograms side-by-side for comparison.
    """
    f1, t1, Sxx1 = spectrogram(signal1, sr, nperseg=1024)
    f2, t2, Sxx2 = spectrogram(signal2, sr, nperseg=1024)

    plt.figure(figsize=(14, 6))

    # Left spectrogram
    plt.subplot(1, 2, 1)
    plt.pcolormesh(t1, f1, 10*np.log10(Sxx1 + 1e-10), shading='gouraud')
    plt.title("Spectrogram — Signal 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")

    # Right spectrogram
    plt.subplot(1, 2, 2)
    plt.pcolormesh(t2, f2, 10*np.log10(Sxx2 + 1e-10), shading='gouraud')
    plt.title("Spectrogram — Signal 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 2) FREQUENCY RESPONSE OF TWO IRs (MAGNITUDE)
# ----------------------------------------------------

def plot_fr(ir1, ir2, sr=48000):
    """
    ir1, ir2: 1D numpy arrays or torch tensors (will be squeezed)
    Plots magnitude response of both IRs on the same graph.
    """

    # Convert from torch if needed
    if hasattr(ir1, "detach"): ir1 = ir1.squeeze().cpu().numpy()
    if hasattr(ir2, "detach"): ir2 = ir2.squeeze().cpu().numpy()

    # FFT frequency axis
    f = np.fft.rfftfreq(len(ir1), 1/sr)

    IR1 = np.fft.rfft(ir1)
    IR2 = np.fft.rfft(ir2)

    mag1 = 20 * np.log10(np.abs(IR1) + 1e-9)
    mag2 = 20 * np.log10(np.abs(IR2) + 1e-9)

    plt.figure(figsize=(12, 5))
    plt.plot(f, mag1, label="IR 1")
    plt.plot(f, mag2, label="IR 2")
    plt.title("Impulse Response Magnitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    samplerate = 48000
    nfft = samplerate
    alias_decay_db = 0.0
    FIR_order = 2**16
    lr = 0.1
    expansion = 2**8
    epochs = 20
    step_size = 200
    step_factor = 0.4

    # Physical room
    dataset_directory = './dataRES'
    room_name = 'MarsioExperimentalStudio3MicSetup2'

    # Loading Dataset 
    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name,
        device = device
    )
    n_M = physical_room.transducer_number['mcs']  # Number of microphones
    n_L = physical_room.transducer_number['lds']  # Number of loudspeakers

    print(f"Number of Mics:", n_M)
    print(f"Number of Loudspeakers:", n_L)

    # Using Random FIRs in the Virtual Room to train for deverb
    virtual_room = random_FIRs(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=FIR_order,
        requires_grad=True,
        device = device
    )

    # Init RES object 
    res = RES(physical_room=physical_room, virtual_room=virtual_room)

    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.backends.cudnn.enabled)

    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU.")

    # Init Model
    # Full system loop, which is closed loop + possible transfer paths  
    model = system.Shell(
        core=res.full_system_(),
        input_layer=dsp.FFT(nfft=nfft),
        output_layer=dsp.iFFT(nfft=nfft)
    )

    model.to(device)

    sys_nat,_,_ = res.system_simulation()
    
    dataset_target = torch.zeros(1, samplerate, 1, device=device)
    dataset_target[0,0:290,0] = sys_nat[0:290,0]
    dataset_target.to(device)

    dataset_input = torch.zeros(1, samplerate, 1, device=device)
    dataset_input[:,0,:] = 1
    dataset_input.to(device)

    print("Dataset Shapes:\n")
    print(f"Input shape:", dataset_input.shape)
    print(f"Target shape:", dataset_target.shape)
    
    dataset = Dataset(
        input = dataset_input,
        target = dataset_target,
        expand = expansion,
        device = device
    )
    
    train_loader, valid_loader = load_dataset(dataset, batch_size=1, split=0.8, shuffle=False)

    trainer = Trainer(
        net=model,
        max_epochs = epochs,
        lr = lr,
        patience_delta = 0.01,
        patience = 5,
        step_size = step_size,
        step_factor = step_factor,
        device = device
    )

    esr = ESRLoss()

    trainer.register_criterion(esr, 2.0 * 1)

    print("Training started...")
    trainer.train(train_loader, valid_loader)
    print("Training ended.")

    _,_,sys_full_opt = res.system_simulation()
    
    speech_opt = convolve("./malespeech.wav", sys_full_opt.squeeze())
    speech_og = convolve("./malespeech.wav", dataset_target.squeeze())
    # ------------------------- Plots -------------------------
    plot_irs_compare(sys_nat, sys_full_opt, samplerate)
    plot_spectrograms(speech_og, speech_opt, 48000)
    plot_fr(speech_og, speech_opt, 48000)