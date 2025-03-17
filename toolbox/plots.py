import matplotlib.pyplot as plt
from matplotlib import mlab
import seaborn as sns
import numpy as np
import torch
from flamo.functional import mag2db, get_magnitude, find_onset


def plot_coupling(rirs):

    rms = torch.sqrt(torch.sum(torch.pow(rirs, 2), dim=(0))/rirs.shape[0])
    norm_rms = rms/torch.max(rms)
    norm_rms_db = 10*torch.log10(norm_rms)
    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    # plt.figure(figsize=(7,6))
    plt.figure()
    image = plt.imshow(norm_rms_db)
    plt.ylabel('Microphone')
    plt.yticks(np.arange(0, rirs.shape[1]), labels=np.arange(1,rirs.shape[1]+1))
    plt.xlabel('Loudspeaker')
    plt.xticks(np.arange(0, rirs.shape[2]), labels=np.arange(1,rirs.shape[2]+1))
    plt.colorbar(mappable=image, label='Magnitude in dB')
    plt.title('Energy coupling between Loudspeakers and Microphones')
    plt.tight_layout()

    plt.show(block=True)


def plot_DRR(rirs, fs):

    drr_lin = torch.zeros(rirs.shape[1], rirs.shape[2])
    for i in range(rirs.shape[1]):
        for j in range(rirs.shape[2]):
            index1 = find_onset(rirs[:,i,j])
            index2 = index1 + int(fs*0.05)
            direct = torch.sum(torch.pow(rirs[index1:index2,i,j], 2))
            reverb = torch.sum(torch.pow(rirs[index2:,i,j], 2))
            drr_lin[i,j] = direct/reverb

    norm_drr = drr_lin/torch.max(drr_lin)
    drr = 10*torch.log10(norm_drr)

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    # plt.figure(figsize=(7,6))
    plt.figure()
    image = plt.imshow(drr)
    plt.ylabel('Microphone')
    plt.yticks(np.arange(0, rirs.shape[1]), labels=np.arange(1,rirs.shape[1]+1))
    plt.xlabel('Loudspeaker')
    plt.xticks(np.arange(0, rirs.shape[2]), labels=np.arange(1,rirs.shape[2]+1))
    plt.colorbar(mappable=image, label='Magnitude in dB')
    plt.title('Direct-to-Reverberant Ratio')
    plt.tight_layout()

    plt.show(block=True)

# def plot_evs_distributions(evs_1: torch.Tensor, evs_2: torch.Tensor, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float, label1: str='Initialized', label2: str='Optimized') -> None:
#     r"""
#     Plot the magnitude distribution of the given eigenvalues.

#         **Args**:
#             evs_init (torch.Tensor): First set of eigenvalues to plot.
#             evs_opt (torch.Tensor): Second set of eigenvalues to plot.
#             fs (int): Sampling frequency.
#             nfft (int): FFT size.
#             label1 (str, optional): Label for the first set of eigenvalues. Defaults to 'Initialized'.
#             label2 (str, optional): Label for the second set of eigenvalues. Defaults to 'Optimized'.
#     """

#     idx1 = int(nfft/fs * lower_f_lim)
#     idx2 = int(nfft/fs * higher_f_lim)
#     evs = mag2db(get_magnitude(torch.cat((evs_1.unsqueeze(-1), evs_2.unsqueeze(-1)), dim=len(evs_1.shape))[idx1:idx2,:,:]))
#     plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
#     plt.figure(figsize=(7,6))
#     ax = plt.subplot(1,1,1)
#     colors = ['tab:blue', 'tab:orange']
#     for i in range(evs.shape[2]):
#         evst = torch.reshape(evs[:,:,i], (evs.shape[0]*evs.shape[1], -1)).squeeze()
#         evst_max = torch.max(evst, 0)[0]
#         ax.boxplot(evst.numpy(), positions=[i], widths=0.7, showfliers=False, notch=True, patch_artist=True, boxprops=dict(edgecolor='k', facecolor=colors[i]), medianprops=dict(color='k'))
#         ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black', facecolors=colors[i])
#     plt.ylabel('Magnitude in dB')
#     plt.xticks([0,1], [label1, label2])
#     plt.xticks(rotation=90)
#     ax.yaxis.grid(True)
#     plt.title(f'Eigenvalue Magnitude Distribution\nbetween {lower_f_lim} Hz and {higher_f_lim} Hz')
#     plt.tight_layout()

    plt.show(block=True)

def plot_evs(evs_init, evs_opt, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float):
    """
    Plot the magnitude distribution of the given eigenvalues.

    Args:
        evs (_type_): _description_
    """

    idx1 = int(nfft/fs * lower_f_lim)
    idx2 = int(nfft/fs * higher_f_lim)
    evs = mag2db(get_magnitude(torch.cat((evs_init.unsqueeze(-1), evs_opt.unsqueeze(-1)), dim=2)[idx1:idx2,:,:]))

    colors = ['xkcd:sky', 'coral', 'coral', "xkcd:mint green", "xkcd:mint green", "xkcd:light magenta", "xkcd:light magenta"]

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    # plt.figure(figsize=(7,6))
    plt.figure()
    ax = plt.subplot(1,1,1)
    for i in range(evs.shape[2]):
        evst = torch.reshape(evs[:,:,i], (evs.shape[0]*evs.shape[1], -1)).squeeze()
        evst_max = torch.max(evst, 0)[0]
        sns.boxplot(evst.numpy(), positions=[i], width=0.7, showfliers=False, patch_artist=True, boxprops=dict(edgecolor='k', facecolor=colors[i]), medianprops=dict(color="k", linewidth=2))
        ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black', facecolors=colors[i])

    ax.yaxis.grid(True)
    plt.xticks([0,1], ['Initialization', 'Optimized'], rotation=60)
    plt.ylabel('Magnitude in dB')
    plt.title(f'Eigenvalue Magnitude Distribution\nbetween {lower_f_lim} Hz and {higher_f_lim} Hz')
    plt.tight_layout()

    plt.show(block=True)

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

    plt.show(block=True)

def plot_ptmr(evs, fs, nfft):
    
    f_axis = torch.linspace(0, fs//2, nfft//2+1)
    evs_peak = torch.max(torch.abs(evs), dim=1)[0]
    evs_mean = torch.mean(torch.abs(evs), dim=1)
    evs_ptmr = evs_peak/evs_mean

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(7,6))
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


def plot_dsps(identity, unitary, firs, modal_reverb, fdn, poletti, fs, nfft):

    n_samples = torch.max(torch.tensor([identity.shape[0], unitary.shape[0], firs.shape[0], modal_reverb.shape[0], fdn.shape[0], poletti.shape[0]]))
    t_axis = torch.linspace(0, n_samples/fs, n_samples)
    y1 = torch.zeros(n_samples,)
    y1[:identity.shape[0]] = identity[:,0,0].squeeze()
    y2 = torch.zeros(n_samples,)
    y2[:unitary.shape[0]] = unitary[:,0,0].squeeze()
    y3 = torch.zeros(n_samples,)
    y3[:firs.shape[0]] = firs[:,0,0].squeeze()
    y4 = torch.zeros(n_samples,)
    y4[:modal_reverb.shape[0]] = modal_reverb[:,0,0].squeeze()
    y5 = torch.zeros(n_samples,)
    y5[:fdn.shape[0]] = fdn[:,0,0].squeeze()
    y6 = torch.zeros(n_samples,)
    y6[:poletti.shape[0]] = poletti[:,0,0].squeeze()

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(7,6))
    plt.subplot(2,1,1)
    
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude in dB')
    # plt.xlim(20,20000)
    # plt.ylim(-60,0)
    # plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)


def plot_raw_evs(evs_init, evs_opt):

    plt.figure()
    plt.plot(mag2db(get_magnitude(evs_init.reshape(evs_init.shape[0]*evs_init.shape[1],1))))
    plt.plot(mag2db(get_magnitude(evs_opt.reshape(evs_opt.shape[0]*evs_opt.shape[1],1))))
    plt.show(block=True)