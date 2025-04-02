import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib import mlab

def plot_boxplot_spectrogram(subplot_spec, fig, tensor_2d, tensor_1d, cmap, vmin, vmax):
    """
    Plots a combined boxplot (top) and spectrogram with colorbar (bottom) within a single subplot.

    Parameters:
        subplot_spec: SubplotSpec -> Subplot specification for the combined plot
        fig: Matplotlib figure -> Figure to which the subplots belong
        tensor_2d: torch.Tensor (2D) -> Data for boxplot (flattened into 1D)
        tensor_1d: torch.Tensor (1D) -> Data for spectrogram
        cmap: str -> Colormap for spectrogram
        vmin, vmax: float -> Color scale limits for spectrogram
    """
    # Create a gridspec within the given subplot_spec
    gs = gridspec.GridSpecFromSubplotSpec(3, 5, subplot_spec=subplot_spec, height_ratios=[0.5, 2, 0.1], hspace=0.7, width_ratios=[4, 0.2, 0.2, 0.2, 0.1])
    
    # Boxplot
    ax_box = fig.add_subplot(gs[0, :4])
    sns.boxplot(x=tensor_2d.flatten().numpy(), ax=ax_box, showfliers=False, patch_artist=True, boxprops=dict(edgecolor='k', facecolor='xkcd:sky'), medianprops=dict(color="k", linewidth=2))
    max_outlier = tensor_2d.flatten().max().item()
    ax_box.scatter([max_outlier], [0], marker="o", s=35, facecolors='lightblue', edgecolors='black', zorder=3)
    ax_box.set_title("Magnitude in dB", fontsize=12)  # Move label above the boxplot
    ax_box.set_yticklabels([])
    
    # Spectrogram
    ax_spec = fig.add_subplot(gs[1, :4])

    spec,f,t = mlab.specgram(tensor_1d.numpy())

    max_val = max(spec.max(), spec.max())
    spec = spec/max_val

    im = ax_spec.pcolormesh(t, f, 10*np.log10(spec), cmap=cmap, vmin=-100, vmax=0)
    ax_spec.set_yscale('log')
    ax_spec.set_ylabel("Frequency in Hz")
    ax_spec.set_xlabel("Time in seconds")
    
    # Colorbar
    cbar_ax = fig.add_subplot(gs[1, 4])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', label="Power in dB")
    
    return im  # Return the image for reference

def plot_grid_boxplot_spectrogram(tensor_pairs, rows, cols, figsize=(12, 8), cmap='viridis'):
    """
    Plots a grid of combined boxplot-spectrogram pairs with row and column labels.

    Parameters:
        tensor_pairs: list of tuples [(2D tensor, 1D tensor), ...] -> Data for each subplot
        rows: int -> Number of rows in grid
        cols: int -> Number of columns in grid
        figsize: tuple -> Figure size
        cmap: str -> Colormap for spectrograms
    """
    plt.rcParams.update({'font.family':'serif', 'font.size':14, 'font.weight':'heavy', 'text.usetex':True})

    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(
        rows,
        cols,
        figure=fig,
        wspace=0.5,  # Space between columns
        hspace=0.5   # Space between rows
    )
    
    # Compute shared color scale for spectrograms
    all_specs = [tensor_1d.numpy() for _, tensor_1d in tensor_pairs]
    vmin = min(np.min(s) for s in all_specs)
    vmax = max(np.max(s) for s in all_specs)
    
    ims = []  # Store images for colorbar reference
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= len(tensor_pairs):
                continue  # Skip if there are fewer pairs than grid cells
            
            tensor_2d, tensor_1d = tensor_pairs[idx]
            subplot_spec = spec[i, j]  # Get the SubplotSpec for this grid cell
            im = plot_boxplot_spectrogram(subplot_spec, fig, tensor_2d, tensor_1d, cmap, vmin, vmax)
            ims.append(im)
    
    # Add row labels using fig.text
    for i in range(rows):
        y = 1 - (i + 0.5) / rows  # Calculate y position for each row
        fig.text(0.02, y, f"Row {i + 1}", va='center', ha='center', fontsize=20, rotation=90)
    
    # Add column labels using fig.text
    for j in range(cols):
        x = (j + 0.5) / cols  # Calculate x position for each column
        fig.text(x, 0.98, f"Column {j + 1}", va='center', ha='center', fontsize=20)
    
    # Adjust the layout to make space for labels
    fig.subplots_adjust(left=0.1, top=0.92, right=0.93, bottom=0.04)
    
    plt.show(block=True)

# Example usage with random tensors
tensor_pairs = [(torch.randn(100, 50), torch.randn(5000)) for _ in range(6)]
plot_grid_boxplot_spectrogram(tensor_pairs, rows=3, cols=2)
