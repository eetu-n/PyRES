# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
import numpy as np
# Torch
import torch
import torch.nn as nn
# Flamo
from flamo import system
from flamo.functional import get_magnitude, get_eigenvalues
from flamo.optimize.loss import mse_loss, masked_mse_loss
from flamo.optimize.utils import generate_partitions

# class evs_mse_loss(mse_loss):
#     def __init__(self, nfft: int, device: str = "cpu"):
#         super().__init__(nfft, device)

#     def forward(self, y_pred, y_true):
#         evs_pred = get_magnitude(get_eigenvalues(y_pred))
#         return self.mse_loss(evs_pred, y_true)

# class masked_mse_loss(nn.Module):

#     def __init__(
#         self,
#         nfft: int,
#         samples_in_partition: int,
#         n_sets: int = 1,   # NOTE: What is the use case of this variable?
#         regenerate_partitions: bool = True,
#         device: str = "cpu",
#     ):
#         super().__init__()
#         self.device = device
#         self.n_samples = samples_in_partition
#         self.n_sets = n_sets
#         self.nfft = nfft
#         self.indices = torch.arange(self.nfft // 2 + 1, device=self.device)
#         self.regenerate_partitions = regenerate_partitions
#         self.partitioning = lambda x: generate_partitions(x, samples_in_partition, n_sets)
#         self.partitions = self.partitioning(self.indices)
#         self.i = -1

#     def forward(self, y_pred, y_true):

#         self.i += 1
#         # generate random mask for sparse sampling
#         if self.i >= self.partitions.shape[0]:
#             self.i = 0
#             if self.regenerate_partitions:
#                 # generate new partitions
#                 self.partitions = self.partitioning(self.indices)
#         mask = self.partitions[self.i]
#         evs_pred = get_magnitude(get_eigenvalues(y_pred[:, mask, :]))
#         evs_true = y_true[:, mask, :]
#         return torch.mean(torch.pow(evs_pred - evs_true, 2))
    
# class bandlimited_masked_mse_loss(masked_mse_loss):

#     def __init__(
#         self,
#         nfft: int,
#         samples_in_partition: int,
#         n_sets: int = 1,
#         regenerate_partitions: bool = True,
#         device: str = "cpu",
#         lowest_f: float = 0,
#         highest_f: float = 0
#     ):
#         super().__init__(nfft, samples_in_partition, n_sets, regenerate_partitions, device)

#         self.lowest_f = lowest_f
#         self.highest_f = highest_f
#         self.min_freq_point = int(lowest_f / (nfft // 2 + 1))
#         self.max_freq_point = int(highest_f / (nfft // 2 + 1))
#         self.indices = torch.arange(self.min_freq_point, self.max_freq_point, device=self.device)
#         self.partitions = self.partitioning(self.indices)


# class indexed_mse_loss(mse_loss):
#     def __init__(self, nfft: int = None, device: str = "cpu", indices: torch.Tensor = None):

#         super().__init__(nfft=nfft, device=device)
#         self.indices = indices

#     def forward(self, y_pred, y_true):
#         """
#         Calculates the mean squared error loss.
#         If :attr:`is_masked` is set to True, the loss is calculated using a masked version of the predicted output. This option is useful to introduce stochasticity, as the mask is generated randomly.

#         **Arguments**:
#             - **y_pred** (torch.Tensor): The predicted output.
#             - **y_true** (torch.Tensor): The target output.

#         Returns:
#             torch.Tensor: The calculated MSE loss.
#         """
#         y_true = y_true.squeeze(-1)[self.indices]
#         y_pred_sum = torch.sum(y_pred[self.indices], dim=-1)
#         return self.mse_loss(y_pred_sum, y_true)
    

class MSE_evs_mod(nn.Module):
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
        super().__init__()

        assert(lowest_f >= 0)
        nyquist = samplerate//2
        assert(highest_f <= nyquist)

        min_freq_point = int(lowest_f/nyquist * freq_points)
        max_freq_point = int(highest_f/nyquist * freq_points)
        # crossover_point = int(crossover_freq/nyquist * freq_points)

        # ratio = (max_freq_point - min_freq_point) / (crossover_point - min_freq_point)
        self.freq_points = max_freq_point - min_freq_point
        self.max_index = self.freq_points

        # self.weights = ( torch.sigmoid(torch.linspace(7, -7*ratio, self.freq_points+min_freq_point)) * 4 ) + 1

        self.iter_num = iter_num
        self.idxs = torch.randperm(self.freq_points) + min_freq_point
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
        idxs = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
        evs_true = y_true[:,idxs,:]
        difference = evs_pred - evs_true
        mask = difference > 0.0
        difference[mask] = difference[mask] * 2
        # weights = self.weights[idxs].unsqueeze(0).unsqueeze(-1).repeat(1,1,evs_true.shape[-1])
        mse = torch.mean(torch.square(torch.abs(difference) )) # * weights
        return mse

    def __get_indexes(self):
        r"""
        Get the indexes of the frequency-point subset.

            **Returns**:
                torch.Tensor: Indexes of the frequency-point subset.
        """
        # Compute indeces
        idx1 = np.min([int(self.interval_count*self.evs_per_iteration), self.max_index-1])
        idx2 = np.min([int((self.interval_count+1) * self.evs_per_iteration), self.max_index])
        idxs = self.idxs[torch.arange(idx1, idx2, dtype=torch.int)]
        # Update interval counter
        self.interval_count = (self.interval_count+1) % (self.iter_num)
        return idxs
    

class MSE_evs_idxs(nn.Module):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int, freqs: torch.Tensor):
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

        freq_axis = torch.linspace(0, samplerate/2, freq_points)
        freqs = freqs.repeat(freq_points, 1)
        freq_axis = freq_axis.unsqueeze(1).repeat(1, freqs.shape[1])
        idxs = torch.argmin(torch.abs(freq_axis - freqs), dim=0)
        self.idxs = torch.cat((idxs-2, idxs-1, idxs, idxs+1, idxs+2), dim=0)
        self.freq_points = len(self.idxs)
        self.iter_num = iter_num

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
        evs_true = y_true[:,self.idxs,:]
        difference = evs_pred - evs_true
        mse = torch.mean(torch.square(torch.abs(difference)))
        return mse

class colorless_reverb(mse_loss):
    def __init__(self, samplerate: int, freq_points: int, freqs: torch.Tensor):
        super().__init__()
        freq_axis = torch.linspace(0, samplerate/2, freq_points)
        freqs = freqs.repeat(freq_points, 1)
        freq_axis = freq_axis.unsqueeze(1).repeat(1, freqs.shape[1])
        idxs = torch.argmin(torch.abs(freq_axis - freqs), dim=0)
        self.idxs = torch.cat((idxs-2, idxs-1, idxs, idxs+1, idxs+2), dim=0)
        self.freq_points = len(self.idxs)

    def forward(self, y_pred, y_target, model):
        processor = system.Shell(core=model.get_core().V_ML)
        mag_response = get_magnitude(processor.get_freq_response(identity=True))
        prediction = mag_response[:,self.idxs,:,:]
        target = torch.ones_like(prediction)

        return self.mse_loss(prediction, target)


# class weighted_bandlimited_evs_mse(bandlimited_evs_mse):
#     def __init__(self, iter_num: int, freq_points: int, samplerate: int, lowest_f: float, crossover_freq: float, highest_f: float):
#         r"""
#         Mean Squared Error (MSE) loss function for Active Acoustics.
#         To reduce computational complexity (i.e. the number of eigendecompositions computed),
#         the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
#         The subset is selected randomly ensuring that all frequency points are considered once and only once.

#             **Args**:
#                 - iter_num (int): Number of iterations per epoch.
#                 - freq_points (int): Number of frequency points.
#         """
#         super().__init__(self, iter_num=iter_num, freq_points=freq_points, samplerate=samplerate, lowest_f=lowest_f, highest_f=highest_f)
#         crossover_point = int(crossover_freq/self.nyquist * freq_points)
#         ratio = (self.max_freq_point - self.min_freq_point) / (crossover_point - self.min_freq_point)
#         self.weights = ( torch.sigmoid(torch.linspace(7, -7*ratio, self.freq_points+self.min_freq_point)) * 4 ) + 1

#     def forward(self, y_pred, y_true):
#         r"""
#         Compute the MSE loss function.
            
#             **Args**:
#                 - y_pred (torch.Tensor): Predicted eigenvalues.
#                 - y_true (torch.Tensor): True eigenvalues.

#             **Returns**:
#                 torch.Tensor: Mean Squared Error.
#         """
#         # Get the indexes of the frequency-point subset
#         idxs = self.__get_indexes()
#         # Get the eigenvalues
#         evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
#         evs_true = y_true[:,idxs,:]
#         difference = evs_pred - evs_true
#         if evs_pred.shape[2] > 4:
#             mask = difference > 0.0
#             difference[mask] = difference[mask] * 3
#         weights = self.weights[idxs].unsqueeze(0).unsqueeze(-1).repeat(1,1,evs_true.shape[-1])
#         mse = torch.mean(torch.square(torch.abs(difference) * weights))
#         return mse
    

# class modified_mse_loss(nn.Module):

#     def __init__(
#         self,
#         nfft: int,
#         device: str = "cpu",
#         partitioned: bool = False,
#         samples_in_partition: int = None,
#         regenerate_partitions: bool = False,
#         bandilited: bool = False,
#         lowest_f: float = None,
#         highest_f: float = None,
#         weighted: bool = False,
#         crossover_f: float = None,
#         indicized: bool = False,
#         indices: torch.Tensor = None
#     ):
#         super().__init__()
#         self.device = device
#         self.n_samples = samples_in_partition
#         self.nfft = nfft

#         if partitioned:
#             self.indices = torch.arange(self.nfft // 2 + 1, device=self.device)
#             self.regenerate_partitions = regenerate_partitions
#             self.partitioning = lambda x: generate_partitions(x, samples_in_partition)
#             self.partitions = self.partitioning(self.indices)
#         self.i = -1

#         self.lowest_f = lowest_f
#         self.highest_f = highest_f
#         self.min_freq_point = int(lowest_f / (nfft // 2 + 1))
#         self.max_freq_point = int(highest_f / (nfft // 2 + 1))
#         self.indices = torch.arange(self.min_freq_point, self.max_freq_point, device=self.device)
#         self.partitions = self.partitioning(self.indices)

#     def forward(self, y_pred, y_true):

#         self.i += 1
#         # generate random mask for sparse sampling
#         if self.i >= self.partitions.shape[0]:
#             self.i = 0
#             if self.regenerate_partitions:
#                 # generate new partitions
#                 self.partitions = self.partitioning(self.indices)
#         mask = self.partitions[self.i]
#         return torch.mean(torch.pow(y_pred[:, mask] - y_true[:, mask], 2))
    
#     def generate_partitions(tensor: torch.Tensor, n_samples: int, seed: Optional[int] = None):
#         if seed is not None:
#             torch.manual_seed(seed)

#         length = len(tensor)
#         n_partitions = length // n_samples

#         # Ensure the tensor length is divisible by N
#         if length % n_samples != 0:
#             print(
#                 "Warning: Tensor length is divisible by n_samples so there will be some samples left out."
#             )

#         partitions_sets = []
#         # Shuffle the tensor
#         shuffled_tensor = tensor[torch.randperm(length)]
#         # Partition the tensor into N equal parts
#         partitions = [
#             shuffled_tensor[i * n_samples : (i + 1) * n_samples]
#             for i in range(n_partitions)
#         ]
#         partitions_sets.append(torch.stack(partitions))

#         partitions_sets = torch.vstack(partitions_sets)
#         return partitions_sets