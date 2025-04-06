import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib import mlab

def plot_combined_figure(tensor_2d_1, tensor_2d_2, tensor_1d_1, tensor_1d_2, cmap='viridis'):
    """
    Produces a figure with:
    - A seaborn boxplot on the left (two boxplots).
    - Two spectrograms stacked vertically on the right, sharing a common colorbar.

    Parameters:
        tensor_2d_1, tensor_2d_2: torch.Tensor (2D) -> Data for the two boxplots.
        tensor_1d_1, tensor_1d_2: torch.Tensor (1D) -> Data for the two spectrograms.
        cmap: str -> Colormap for the spectrograms.
    """
    plt.rcParams.update({'font.family': 'serif', 'font.size': 20, 'font.weight': 'heavy', 'text.usetex': True})

    # Create the figure and gridspec
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 5, width_ratios=[0.7, 0.5, 2, 0.1, 0.1], height_ratios=[1, 1], wspace=0, hspace=0.3)

    # Left subplot: Boxplot
    ax_box = fig.add_subplot(gs[:, 0])  # Use both rows for the boxplot
    data = [tensor_2d_1.flatten().numpy(), tensor_2d_2.flatten().numpy()]
    sns.boxplot(data=data, ax=ax_box, showfliers=False, palette="pastel", boxprops=dict(edgecolor='k'),
                medianprops=dict(color="k", linewidth=2))
    ax_box.set_title("Boxplot", fontsize=16)
    ax_box.set_xticklabels(["Box 1", "Box 2"])
    ax_box.set_ylabel("Magnitude in dB")

    # Right subplot: Spectrograms
    ax_spec1 = fig.add_subplot(gs[0, 2])  # Top spectrogram
    ax_spec2 = fig.add_subplot(gs[1, 2])  # Bottom spectrogram

    # Compute spectrograms
    spec1, f1, t1 = mlab.specgram(tensor_1d_1.numpy(), NFFT=256, Fs=1, noverlap=128)
    spec2, f2, t2 = mlab.specgram(tensor_1d_2.numpy(), NFFT=256, Fs=1, noverlap=128)

    # Normalize spectrograms
    max_val = max(spec1.max(), spec2.max())
    spec1 /= max_val
    spec2 /= max_val

    # Plot spectrograms
    im1 = ax_spec1.pcolormesh(t1, f1, 10 * np.log10(spec1), cmap=cmap, vmin=-100, vmax=0)
    im2 = ax_spec2.pcolormesh(t2, f2, 10 * np.log10(spec2), cmap=cmap, vmin=-100, vmax=0)

    # Set labels
    ax_spec1.set_ylabel("Frequency (Hz)")
    ax_spec1.set_title("Spectrogram 1", fontsize=14)
    ax_spec2.set_ylabel("Frequency (Hz)")
    ax_spec2.set_xlabel("Time (s)")
    ax_spec2.set_title("Spectrogram 2", fontsize=14)

    # Common colorbar
    cbar_ax = fig.add_subplot(gs[:, 4])  # Use both rows for the colorbar
    fig.colorbar(im1, cax=cbar_ax, orientation='vertical', label="Power (dB)")

    fig.subplots_adjust(left=0.08, right=0.89, top=0.95, bottom=0.12)

    # Show the plot
    plt.show(block=True)

# Example usage with random tensors
tensor_2d_1 = torch.randn(100, 50)
tensor_2d_2 = torch.randn(100, 50)
tensor_1d_1 = torch.randn(5000)
tensor_1d_2 = torch.randn(5000)

plot_combined_figure(tensor_2d_1, tensor_2d_2, tensor_1d_1, tensor_1d_2)