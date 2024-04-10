import torch.nn as nn
import torch

from CONFIG import DTYPE_TORCH


class AUTOENCODER(nn.Module):
    """Conditions need to be dim 1, number of channels dim 2, data/time points dim 3
    Example array with 3 conditions, 2 channels and 4 points per channel:
    np.array(
        [
            [[1, 2, 3, 4], [1, 2, 3, 4]],
            [[2, 3, 4, 5], [2, 3, 4, 5]],
            [[0, 1, 2, 3], [0, 1, 2, 3]]
        ],
        dtype="float64")
    """

    def __init__(self, in_dim: int):
        super().__init__()

        self.in_dim = in_dim

        self.process = nn.Sequential(
            nn.Linear(self.in_dim, 20, dtype=DTYPE_TORCH),  # Layer 1 (in -> 20)
            nn.ReLU(),
            nn.Linear(20, 10, dtype=DTYPE_TORCH),  # Layer 4 (20 -> 10)
            nn.ReLU(),
            nn.Linear(10, 1, dtype=DTYPE_TORCH),  # Layer 5 (10 -> 1)
            nn.ReLU(),
            nn.Linear(1, 10, dtype=DTYPE_TORCH),  # Layer 5 (1 -> 10)
            nn.ReLU(),
            nn.Linear(10, 20, dtype=DTYPE_TORCH),  # Layer 4 (10 -> 20)
            nn.ReLU(),
            nn.Linear(20, self.in_dim, dtype=DTYPE_TORCH),  # Layer 1 (20 -> in)
            nn.ReLU(),
        )

    @staticmethod
    def reshape_data(data):
        """Assumes that the data is 5 dimensional"""
        data_copy = data[:]
        data_copy = data_copy.transpose(3, 4)
        data_copy = data_copy.reshape(
            data_copy.shape[0]
            * data_copy.shape[1]
            * data_copy.shape[2]
            * data_copy.shape[3],
            data_copy.shape[4],
        )
        return data_copy

    @staticmethod
    def unshape_data(data, shape):
        data_copy = data[:]
        data_copy = data_copy.reshape(shape[0], shape[1], shape[2], shape[4], shape[3])
        return data_copy.transpose(3, 4)

    def forward(self, data):
        new_data = self.reshape_data(data)

        return self.unshape_data(self.process(new_data), data.shape)
