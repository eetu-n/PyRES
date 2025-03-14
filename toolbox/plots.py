import matplotlib.pyplot as plt
from matplotlib import mlab
import seaborn as sns
import numpy as np
import torch
from flamo.functional import mag2db, get_magnitude

# PLOTS THAT I WANT TO HAVE:
# Physical room:
# - Mean squared RIR wiht variance over time for given RIRs.
# - Mean magnitude RTF with variance over frequency for given RTFs.
# Virtual room:
# - Use Shell methods to get RIRs and RTFs and plot them.
# Eigenvalues:
# - Eigenvalue magnitude distribution.
# - Peak-to-mean over frequency -> coloration coefficient.
# Auralization:
# - IR spectrogram.


def plot_coupling(rirs):
    norm_val = torch.norm(rirs, 'fro')

    rms = torch.sum(torch.pow(rirs, 2), dim=(0))/rirs.shape[0]
    plt.figure()
    image = plt.imshow(rms)
    plt.ylabel('Microphone')
    plt.yticks(np.arange(0, rirs.shape[1]), labels=np.arange(1,rirs.shape[1]+1))
    plt.xlabel('Loudspeaker')
    plt.xticks(np.arange(0, rirs.shape[2]), labels=np.arange(1,rirs.shape[2]+1))
    plt.colorbar(mappable=image)

    # new_rirs = rirs/rms
    # new_rirs = new_rirs/torch.norm(new_rirs, 'fro') * norm_val
    # rms = torch.sqrt(torch.sum(torch.pow(new_rirs, 2), dim=(0))/rirs.shape[0])
    # plt.figure()
    # image = plt.imshow(rms)
    # plt.ylabel('Microphone')
    # plt.yticks(np.arange(0, rirs.shape[1]), labels=np.arange(1,rirs.shape[1]+1))
    # plt.xlabel('Loudspeaker')
    # plt.xticks(np.arange(0, rirs.shape[2]), labels=np.arange(1,rirs.shape[2]+1))
    # plt.colorbar(mappable=image)
    plt.show(block=True)



def plot_evs_distributions(evs_1: torch.Tensor, evs_2: torch.Tensor, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float, label1: str='Initialized', label2: str='Optimized') -> None:
    r"""
    Plot the magnitude distribution of the given eigenvalues.

        **Args**:
            evs_init (torch.Tensor): First set of eigenvalues to plot.
            evs_opt (torch.Tensor): Second set of eigenvalues to plot.
            fs (int): Sampling frequency.
            nfft (int): FFT size.
            label1 (str, optional): Label for the first set of eigenvalues. Defaults to 'Initialized'.
            label2 (str, optional): Label for the second set of eigenvalues. Defaults to 'Optimized'.
    """

    idx1 = int(nfft/fs * lower_f_lim)
    idx2 = int(nfft/fs * higher_f_lim)
    evs = mag2db(get_magnitude(torch.cat((evs_1.unsqueeze(-1), evs_2.unsqueeze(-1)), dim=len(evs_1.shape))[idx1:idx2,:,:]))
    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(7,6))
    ax = plt.subplot(1,1,1)
    colors = ['tab:blue', 'tab:orange']
    for i in range(evs.shape[2]):
        evst = torch.reshape(evs[:,:,i], (evs.shape[0]*evs.shape[1], -1)).squeeze()
        evst_max = torch.max(evst, 0)[0]
        ax.boxplot(evst.numpy(), positions=[i], widths=0.7, showfliers=False, notch=True, patch_artist=True, boxprops=dict(edgecolor='k', facecolor=colors[i]), medianprops=dict(color='k'))
        ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black', facecolors=colors[i])
    plt.ylabel('Magnitude in dB')
    plt.xticks([0,1], [label1, label2])
    plt.xticks(rotation=90)
    ax.yaxis.grid(True)
    plt.title(f'Eigenvalue Magnitude Distribution\nbetween {lower_f_lim} Hz and {higher_f_lim} Hz')
    plt.tight_layout()

def plot_evs(evs, *kwargs):
        """
        Plot the magnitude distribution of the given eigenvalues.

        Args:
            evs (_type_): _description_
        """
        plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
        plt.figure(figsize=(7,6))
        ax = plt.subplot(1,1,1)
        for i in range(evs.shape[2]):
            evst = torch.reshape(evs[:,:,:,i], (evs.shape[1]*evs.shape[2], -1)).squeeze()
            evst_max = torch.max(evst, 0)[0]
            sns.boxplot(evst.numpy(), positions=[i], width=0.7, showfliers=False)
            ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black')

        plt.xticks([0,1], ['Initialization', 'Optimized'])
        plt.xticks(rotation=90)
        ax.yaxis.grid(True)
        plt.tight_layout()

        plt.show()

def plot_spectrograms(y_1: torch.Tensor, y_2: torch.Tensor, fs: int, nfft: int=2**10, noverlap: int=2**8, label1='Initialized', label2='Optimized', title='System Impulse Response Spectrograms') -> None:
    r"""
    Plot the spectrograms of the system impulse responses at initialization and after optimization.
    
        **Args**:
            - y_1 (torch.Tensor): First signal to plot.
            - y_2 (torch.Tensor): Second signal to plot.
            - fs (int): Sampling frequency.
            - nfft (int, optional): FFT size. Defaults to 2**10.
            - label1 (str, optional): Label for the first signal. Defaults to 'Initialized'.
            - label2 (str, optional): Label for the second signal. Defaults to 'Optimized'.
            - title (str, optional): Title of the plot. Defaults to 'System Impulse Response Spectrograms'.
    """
    Spec_init,f,t = mlab.specgram(y_1.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)
    Spec_opt,_,_ = mlab.specgram(y_2.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=noverlap)

    max_val = max(Spec_init.max(), Spec_opt.max())
    Spec_init = Spec_init/max_val
    Spec_opt = Spec_opt/max_val
    

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    fig,axes = plt.subplots(2,1, sharex=False, sharey=True, figsize=(7,5), constrained_layout=True)
    
    plt.subplot(2,1,1)
    plt.pcolormesh(t, f, 10*np.log10(Spec_init), cmap='magma', vmin=-100, vmax=0)
    plt.ylim(20, 20000)
    plt.yscale('log')
    plt.title(label1)

    plt.subplot(2,1,2)
    im = plt.pcolormesh(t, f, 10*np.log10(Spec_opt), cmap='magma', vmin=-100, vmax=0)
    plt.ylim(20, 20000)
    plt.yscale('log')
    plt.title(label2)

    fig.supxlabel('Time in seconds')
    fig.supylabel('Frequency in Hz')
    fig.suptitle(title)

    cbar = fig.colorbar(im, ax=axes[:], aspect=20)
    cbar.set_label('Magnitude in dB')
    ticks = np.arange(-100, 1, 20)
    cbar.ax.set_ylim(-100, 0)
    cbar.ax.set_yticks(ticks, ['-100','-80','-60','-40','-20','0'])

def plot_ptmr(evs, fs, nfft):
    
    f_axis = torch.linspace(0, fs//2, nfft//2+1)
    evs_peak = torch.max(torch.abs(evs), dim=1)[0]
    evs_mean = torch.mean(torch.abs(evs), dim=1)
    evs_ptmr = evs_peak/evs_mean

    plt.figure()
    plt.plot(f_axis, mag2db(evs_peak))
    plt.plot(f_axis, mag2db(evs_mean))
    plt.plot(f_axis, mag2db(evs_ptmr))
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Magnitude in dB')
    plt.ylim(-50,10)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.legend(['Peak value', 'Mean value', 'Peak-to-mean ratio'])
    # plt.ylim(-20,30)
    # plt.xlim(20,20000)
    # plt.xscale('log')
    # plt.grid()
    plt.tight_layout()
    plt.show(block=True)

def plot_line(rirs, fs, nfft):
    import pandas

    rtfs = torch.fft.rfft(rirs, nfft, dim=0)
    data = (pandas.DataFrame(data = mag2db(torch.abs(rtfs.view(rtfs.shape[0]*rtfs.shape[1]*rtfs.shape[2]))).numpy()))
    f_axis = torch.linspace(0, fs//2, nfft//2+1)
    sns.relplot(data=data, kind='line')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Magnitude in dB')
    # plt.ylim(-50,0)
    # plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)

def plot_I_want(rirs,fs,nfft):

    # rtfs = torch.fft.rfft(rirs, nfft, dim=0).view(nfft//2+1, -1)
    f_axis = torch.linspace(0, fs//2, nfft//2+1)
    # y = mag2db(torch.abs(rtfs))
    y = torch.square(rirs.view(rirs.shape[0],-1))
    t_axis = torch.linspace(0, rirs.shape[0]/fs, rirs.shape[0])
    y1 = torch.max(y, dim=1)[0]
    y2 = torch.min(y, dim=1)[0]

    plt.figure()
    plt.fill_between(t_axis, y1, y2, color='tab:blue', alpha=0.5)
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude in dB')
    # plt.xlim(20,20000)
    # plt.ylim(-60,0)
    # plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)


def plot_stuff(samplerate, nfft, rtfs, fl_rtfs, evs):

    f_axis = torch.linspace(0, samplerate//2, nfft//2+1)
    rtfs_peak = torch.max(torch.max(torch.abs(rtfs), dim=2)[0], dim=1)[0]
    rtfs_mean = torch.mean(torch.abs(rtfs), dim=(1,2))
    rtfs_ptmr = rtfs_peak / rtfs_mean
    if len(fl_rtfs.shape) < 3:
        fl_rtfs = fl_rtfs.unsqueeze(-1)
    fl_rtfs_peak = torch.max(torch.max(torch.abs(fl_rtfs), dim=2)[0], dim=1)[0]
    fl_rtfs_mean = torch.mean(torch.abs(fl_rtfs), dim=(1,2))
    fl_rtfs_ptmr = fl_rtfs_peak/fl_rtfs_mean
    evs_peak = torch.max(torch.abs(evs), dim=1)[0]
    evs_mean = torch.mean(torch.abs(evs), dim=1)
    evs_ptmr = evs_peak/evs_mean

    plt.figure()
    plt.subplot(231)
    plt.plot(f_axis, mag2db(rtfs_peak), label='Maximum')
    plt.plot(f_axis, mag2db(rtfs_mean), label='Mean')
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.title('Room transfer functions - magnitude')
    plt.subplot(234)
    plt.plot(f_axis, mag2db(rtfs_ptmr))
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.subplot(232)
    plt.plot(f_axis, mag2db(fl_rtfs_peak))
    plt.plot(f_axis, mag2db(fl_rtfs_mean))
    plt.ylim(-50,0)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.title('Feedback-loop transfer functions - magnitude')
    plt.subplot(235)
    plt.plot(f_axis, mag2db(fl_rtfs_ptmr))
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.subplot(233)
    plt.plot(f_axis, mag2db(evs_peak))
    plt.plot(f_axis, mag2db(evs_mean))
    plt.ylim(-50,0)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.title('Eigenvalues - magnitude')
    plt.subplot(236)
    plt.plot(f_axis, mag2db(evs_ptmr))
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)