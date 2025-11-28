
import numpy as np
import scipy.io.wavfile as wavfile 
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.signal import spectrogram

import torch 
from PyRES.plots import plot_irs_compare
from PyRES.rds import RDS

def convolve(signal_path, ir):

    sr, data = wavfile.read(signal_path)

    y = fftconvolve(data, ir, mode='full')
    y = y / np.max(np.abs(y) + 1e-9)
    wavfile.write("./malespeech_conv.wav", sr, y)
    
    return y, data

def plot_mag_diff(sig1, sig2, sr=48000):
    sig1 = np.asarray(sig1).squeeze()
    sig2 = np.asarray(sig2).squeeze()

    n = min(len(sig1), len(sig2))
    sig1 = sig1[:n]
    sig2 = sig2[:n]

    f = np.fft.rfftfreq(n, 1/sr)

    S1 = np.fft.rfft(sig1)
    S2 = np.fft.rfft(sig2)

    mag1 = 20 * np.log10(np.abs(S1) + 1e-9)
    mag2 = 20 * np.log10(np.abs(S2) + 1e-9)

    diff = mag1 - mag2

    plt.figure(figsize=(12,5))
    plt.plot(f, diff)
    plt.title("Magnitude Difference (Sig1 – Sig2)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Difference (dB)")
    plt.grid(True)
    plt.xlim([0, 20000])
    plt.tight_layout()
    plt.show()

def plot_spectrograms(signal1, signal2, sr):
    signal1 = np.asarray(signal1).squeeze()
    signal2 = np.asarray(signal2).squeeze()

    L = max(len(signal1), len(signal2))
    if len(signal1) < L:
        signal1 = np.pad(signal1, (0, L - len(signal1)))
    if len(signal2) < L:
        signal2 = np.pad(signal2, (0, L - len(signal2)))

    f1, t1, Sxx1 = spectrogram(signal1, sr, nperseg=4096, noverlap=2048)
    f2, t2, Sxx2 = spectrogram(signal2, sr, nperseg=4096, noverlap=2048)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.pcolormesh(t1, f1, 10*np.log10(Sxx1 + 1e-10), shading='gouraud')
    plt.title("Spectrogram — Signal 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")

    plt.subplot(1, 2, 2)
    plt.pcolormesh(t2, f2, 10*np.log10(Sxx2 + 1e-10), shading='gouraud')
    plt.title("Spectrogram — Signal 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")

    plt.tight_layout()
    plt.show()


def plot_fr(ir1, ir2, sr=48000):
    ir1 = np.asarray(ir1).squeeze()
    ir2 = np.asarray(ir2).squeeze()

    L = max(len(ir1), len(ir2))
    if len(ir1) < L:
        ir1 = np.pad(ir1, (0, L - len(ir1)))
    if len(ir2) < L:
        ir2 = np.pad(ir2, (0, L - len(ir2)))

    f = np.fft.rfftfreq(L, 1/sr)

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
    plt.xlim([0, 20000])
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.set_default_device(device)
    
    fs = 48000

    rds = RDS(
        fs = fs,
        FIR_order = 2**16,
        expansion = 2**10,
        epochs = 10,
        room_name = "MarsioExperimentalStudio3MicSetup2",
        lds_idx = [0, 5],
        mcs_idx = [2, 3],
        device = torch.get_default_device()
    )

    print("Training started...")
    rds.train()
    print("Training ended.")
    # ------------------------- Plots -------------------------
    sys_nat,_,sys_full_opt = rds.res.system_simulation()

    _,_,sys_full_opt = rds.res.system_simulation()
    
    speech_opt, speech = convolve("./malespeech.wav", sys_full_opt.squeeze())

    print("speech:", type(speech), np.shape(speech))
    print("speech_opt:", type(speech_opt), np.shape(speech_opt))
    print("fs:", type(fs), fs)
    # ------------------------- Plots -------------------------
    plot_irs_compare(sys_nat, sys_full_opt, fs)
    plot_spectrograms(speech, speech_opt, fs)
    plot_fr(speech, speech_opt, fs)
    plot_mag_diff(speech, speech_opt, fs)