import numpy as np
import torch
import torch.nn as nn
from flamo.functional import (
    get_magnitude,
    get_eigenvalues
)

class evs_mse(nn.Module):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
        """
        super().__init__()

        self.freq_points = freq_points
        self.max_index = self.freq_points

        self.iter_num = iter_num
        self.idxs = torch.randperm(self.freq_points)
        self.evs_per_iteration = torch.ceil(torch.tensor(self.freq_points / self.iter_num, dtype=torch.float))
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                torch.Tensor: Mean Squared Error.
        """
        # Get the indexes of the frequency-point subset
        idxs = self.__get_indexes(max_index=self.max_index)
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
        evs_true = y_true[:,idxs,:]
        mse = torch.mean(torch.square(torch.abs(evs_pred - evs_true)))
        return mse

    def __get_indexes(self, max_index: int=None):
        r"""
        Get the indexes of the frequency-point subset.

            **Returns**:
                torch.Tensor: Indexes of the frequency-point subset.
        """
        # Compute indeces
        idx1 = np.min([int(self.interval_count*self.evs_per_iteration), max_index-1])
        idx2 = np.min([int((self.interval_count+1) * self.evs_per_iteration), max_index])
        idxs = self.idxs[torch.arange(idx1, idx2, dtype=torch.int)]
        # Update interval counter
        self.interval_count = (self.interval_count+1) % (self.iter_num)
        return idxs

class bandlimited_evs_mse(evs_mse):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int, lowest_f: float, highest_f: float):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
        """
        super().__init__(iter_num=iter_num, freq_points=freq_points, samplerate=samplerate)

        assert(lowest_f >= 0)
        self.nyquist = samplerate//2
        assert(highest_f <= self.nyquist)

        self.min_freq_point = int(lowest_f/self.nyquist * freq_points)
        self.max_freq_point = int(highest_f/self.nyquist * freq_points)
        self.freq_points = self.max_freq_point - self.min_freq_point
        self.max_index = self.freq_points

        self.iter_num = iter_num
        self.idxs = torch.randperm(self.freq_points) + self.min_freq_point
        self.evs_per_iteration = torch.ceil(torch.tensor(self.freq_points / self.iter_num, dtype=torch.float))
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                torch.Tensor: Mean Squared Error.
        """
        # Get the indexes of the frequency-point subset
        idxs = self.__get_indexes(max_index=self.max_freq_point)
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
        evs_true = y_true[:,idxs,:]
        mse = torch.mean(torch.square(torch.abs(evs_pred - evs_true)))
        return mse
    
class weighted_bandlimited_evs_mse(bandlimited_evs_mse):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int, lowest_f: float, crossover_freq: float, highest_f: float):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
        """
        super().__init__(self, iter_num=iter_num, freq_points=freq_points, samplerate=samplerate, lowest_f=lowest_f, highest_f=highest_f)
        crossover_point = int(crossover_freq/self.nyquist * freq_points)
        ratio = (self.max_freq_point - self.min_freq_point) / (crossover_point - self.min_freq_point)
        self.weights = ( torch.sigmoid(torch.linspace(7, -7*ratio, self.freq_points+self.min_freq_point)) * 4 ) + 1

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                torch.Tensor: Mean Squared Error.
        """
        # Get the indexes of the frequency-point subset
        idxs = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
        evs_true = y_true[:,idxs,:]
        difference = evs_pred - evs_true
        if evs_pred.shape[2] > 4:
            mask = difference > 0.0
            difference[mask] = difference[mask] * 3
        weights = self.weights[idxs].unsqueeze(0).unsqueeze(-1).repeat(1,1,evs_true.shape[-1])
        mse = torch.mean(torch.square(torch.abs(difference) * weights))
        return mse
    
    
class minimize_evs_mod(nn.Module):
    def __init__(self, iter_num: int, idxs: torch.Tensor):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
        """
        super().__init__()

        self.iter_num = iter_num
        self.idxs = idxs

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                torch.Tensor: Mean Squared Error.
        """
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,self.idxs,:,:]))
        mse = torch.mean(torch.square(evs_pred))
        return mse
    
class preserve_energy(nn.Module):
    def __init__(self, idxs):
        super().__init__()
        self.idxs = idxs
    def forward(self, y_pred, y_target, model):
        freq_response = model.F_MM._Shell__core.MR.freq_response
        return torch.mean(torch.pow(torch.abs(freq_response[self.idxs])-torch.abs(y_target.squeeze()[self.idxs]), 2))