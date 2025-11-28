
import numpy as np
import scipy.io.wavfile as wavfile 
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.signal import spectrogram

import torch 
from PyRES.plots import plot_irs_compare
from PyRES.rds import RDS

def convolve(signal_path, ir, output_path="./malespeech_conv.wav"):
    try:
        if isinstance(signal_path, str):
            sr, data = wavfile.read(signal_path)
            print(f"Input: SR={sr}, Shape={data.shape}, dtype={data.dtype}")
        else:
            data = signal_path
            sr = 48000
            print(f"Input: Direct array - SR={sr}, Shape={data.shape}, dtype={data.dtype}")
        
        print(f"Input: SR={sr}, Shape={data.shape}, dtype={data.dtype}")
        print(f"IR: Shape={ir.shape}, dtype={ir.dtype}")

        if hasattr(ir, 'detach'):  
            ir = ir.detach().cpu().numpy()
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            print("Converted input to mono")
            
        if len(ir.shape) > 1:
            ir = np.mean(ir, axis=1)
            print("Converted IR to mono")
        
        data = data.astype(np.float32)
        ir = ir.astype(np.float32)

        data_max = np.max(np.abs(data))
        ir_max = np.max(np.abs(ir))
        
        if data_max > 0:
            data = data / data_max
        if ir_max > 0:
            ir = ir / ir_max
        
        print(f"Normalized: data max={np.max(np.abs(data)):.3f}, IR max={np.max(np.abs(ir)):.3f}")
        
        y = fftconvolve(data, ir, mode='full')

        y_max = np.max(np.abs(y))
        if y_max > 0:
            y = y / y_max
        
        print(f"Output: max={np.max(np.abs(y)):.3f}, length={len(y)}")

        y_int16 = np.int16(y * 32767)

        if output_path is not None:
            y_int16 = np.int16(y * 32767)
            wavfile.write(output_path, sr, y_int16)
            print(f"Output written to: {output_path}")
        else:
            print("Skipping file write (output_path is None)")
        
        return y, data
        
    except Exception as e:
        print(f"Error in convolution: {e}")
        return None, None

def plot_mag_diff_third_octave(sig1, sig2, sr=48000):
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

    center_freqs = []
    f0 = 1000  
    for i in range(-16, 13):  
        center_freqs.append(f0 * (2 ** (i / 3)))
    center_freqs = np.array(center_freqs)
    
    center_freqs = center_freqs[(center_freqs >= 20) & (center_freqs <= 20000)]
    
    band_edges_low = center_freqs / (2 ** (1/6))
    band_edges_high = center_freqs * (2 ** (1/6))
    
    band_diffs = []
    band_centers_used = []
    
    for i, (flow, fcenter, fhigh) in enumerate(zip(band_edges_low, center_freqs, band_edges_high)):
        mask = (f >= flow) & (f <= fhigh)
        if np.sum(mask) > 0:  
            band_diff = np.mean(diff[mask])
            band_diffs.append(band_diff)
            band_centers_used.append(fcenter)
    
    band_diffs = np.array(band_diffs)
    band_centers_used = np.array(band_centers_used)
    
    plt.figure(figsize=(12, 5))
    
    plt.plot(f, diff, color='black', alpha=0.5, label='Spectral difference')
    
    plt.bar(band_centers_used, band_diffs, 
            width=band_edges_high - band_edges_low,
            alpha=0.7, color='blue', edgecolor='darkblue', label='1/3 Octave bands')
    
    plt.title("Magnitude Difference in 1/3 Octave Bands")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Difference (dB)")
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xlim([20, 20000])
    
    major_ticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    plt.xticks(major_ticks, [f'{f:.1f}' if f < 1000 else f'{f/1000:.0f}k' for f in major_ticks])
    plt.ylim([-10, 60])
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    plt.title("Magnitude Difference")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Difference (dB)")
    plt.grid(True)
    plt.xlim([0, 20000])
    plt.ylim([-10, 60])
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

    f1, t1, Sxx1 = spectrogram(signal1, sr, nperseg=2048, noverlap=1024)
    f2, t2, Sxx2 = spectrogram(signal2, sr, nperseg=2048, noverlap=1024)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.pcolormesh(t1, f1, 10*np.log10(Sxx1 + 1e-10), shading='gouraud')
    plt.title("Spectrogram — Anechoic Speech")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")

    plt.subplot(1, 2, 2)
    plt.pcolormesh(t2, f2, 10*np.log10(Sxx2 + 1e-10), shading='gouraud')
    plt.title("Spectrogram — Convolved Speech")
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
    plt.plot(f, mag1, label="Anechoic Speech")
    plt.plot(f, mag2, label="Convolved Speech")
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

    n_samples = int(fs * 1.0)
    t = np.linspace(0, 1.0, n_samples, endpoint=False)
    
    white_noise = np.random.normal(0, 1, n_samples)
    
    speech_opt, speech = convolve(white_noise, sys_full_opt.squeeze(), None)
    speech_nat, speech = convolve(white_noise, sys_nat.squeeze(), None)

    # ------------------------- Plots -------------------------
    plot_irs_compare(sys_nat, sys_full_opt, fs)
    #plot_spectrograms(speech, speech_opt, fs)
    #plot_fr(speech, speech_opt, fs)
    #plot_mag_diff(speech, speech_opt, fs)
    plot_mag_diff_third_octave(speech_nat, speech_opt, fs)