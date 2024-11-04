import torch
import torch.nn as nn

from flamo.functional import (
    get_magnitude,
    get_eigenvalues
)

class MSE_evs(nn.Module):
    """
    Mean Squared Error (MSE) loss function for eigenvalues.
    To reduce computational complexity, the loss is computed only on a subset of the eigenvalues at each iteration of an epoch.
    """
    def __init__(self, ds_size, freq_points):
        """
        Initialize the MSE loss function for eigenvalues.

        Args:
            ds_size (_type_): _description_
            freq_points (_type_): _description_
        """
        super().__init__()
        # The number of intervals matches the dataset size
        self.interval_idxs = torch.randperm(ds_size)
        # The number of eigenvalues common to two adjacent intervals
        self.overlap = torch.tensor(500, dtype=torch.int)
        # The number of eigenvalues per interval
        int_width = torch.max(torch.tensor([freq_points//ds_size, 2400], dtype=torch.int))
        self.evs_numb = torch.tensor(int_width + self.overlap, dtype=torch.int)
        assert self.evs_numb < freq_points, "The number of eigenvalues per interval is too large."
        # Auxiliary variable to go through the intervals
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        # Get the indexes of the current interval
        idx1, idx2 = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idx1:idx2,:,:]))
        evs_true = y_true[:,idx1:idx2,:]
        # Compute the loss
        MSE = torch.mean(torch.square(evs_pred - evs_true))
        return MSE

    def __get_indexes(self):
        # Don't exceed the size of the tensors
        max_index = self.evs_numb * len(self.int_idx)
        min_index = 0
        # Compute indeces
        idx1 = torch.max(torch.tensor( [min_index, self.int_idx[self.i]*self.evs_numb - self.overlap], dtype=torch.int))
        idx2 = torch.min(torch.tensor( [(self.int_idx[self.i]+1)*self.evs_numb - self.overlap, max_index], dtype=torch.int))
        # Update interval counter
        self.i = (self.i+1) % len(self.int_idx)
        return idx1, idx2
    

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
    
class preserve_reverb_energy_mod(nn.Module):
    def __init__(self, idxs):
        super().__init__()
        self.idxs = idxs
    def forward(self, y_pred, y_target, model):
        freq_response = model.F_MM._Shell__core.MR.freq_response
        return torch.mean(torch.pow(torch.abs(freq_response[self.idxs])-torch.abs(y_target.squeeze()[self.idxs]), 2))