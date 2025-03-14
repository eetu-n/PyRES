import numpy as np
import torch


def system_equalization_curve(
        evs: torch.Tensor,
        fs: int,
        nfft: int,
        f_crossover: float=None
    ) -> torch.Tensor:
        f"""
        Computes the system equalization curve: flat response below the crossover frequency and moving average of the RTFs above the crossover frequency.
        
            **Args**:
                evs (torch.Tensor): Open loop eigenvalues. shape = (samples, n_M).
                fs (int): Sampling frequency [Hz].
                nfft (int): Number of frequency bins.
                f_crossover (float, optional): Crossover frequency [Hz]. Defaults to None.
            
            **Returns**:
                torch.Tensor: The system equalization curve.
        """
        
        # frequency samples
        freqs = torch.linspace(0, fs/2, nfft//2+1)

        # Compute RTFs
        mean_evs = torch.mean(torch.abs(evs), dim=(1))

        if f_crossover is not None:
            # Divide target between left and right of crossover frequency
            index_crossover = torch.argmin(torch.abs(freqs - f_crossover))
            left_interval = torch.arange(0, index_crossover+1)
            right_interval = torch.arange(index_crossover, mean_evs.shape[0])

            # Left target: horizontal line at mean RTFs value
            scaling_factor = torch.mean(mean_evs[left_interval])
            target_left = scaling_factor * torch.ones(index_crossover,)

            # Right target: moving average of RTFs values
            smooth_window_length = right_interval.shape[0]//6
            smooth_evs = torch.tensor(np.convolve(mean_evs[right_interval], np.ones(smooth_window_length)/smooth_window_length, mode='valid'))
            pre = torch.ones(smooth_window_length//2, 1) * smooth_evs[0]
            post = torch.ones(smooth_window_length//2, 1) * smooth_evs[-1]
            target_right = torch.cat((pre, smooth_evs, post), dim=0)

            # Concatenate left and right targets
            target = torch.cat((target_left, target_right), dim=0)
        else:
            # Horizontal line at mean RTFs value
            scaling_factor = torch.mean(mean_evs)
            target = scaling_factor * torch.ones(mean_evs.shape[0],)
        
        return target