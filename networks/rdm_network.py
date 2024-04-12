import torch
import torch.nn as nn
import numpy as np

from CONFIG import DTYPE_TORCH

from util.data import create_rdms


class RDM_MLP(nn.Module):
    """Conditions need to be dim 1, number of channels dim 2, data/time points dim 3
    Example array with 3 conditions, 2 channels and 4 points per channel:
    np.array(
        [
            [[1, 2, 3, 4], [1, 2, 3, 4]],
            [[2, 3, 4, 5], [2, 3, 4, 5]],
            [[0, 1, 2, 3], [0, 1, 2, 3]]
        ],
        dtype="float32")
    """

    def __init__(self, in_dim: int):
        super().__init__()

        self.in_dim = in_dim

        self.process = nn.Sequential(
            nn.Linear(self.in_dim, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 10, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(10, 1, dtype=DTYPE_TORCH),
            nn.ReLU(),
        )

    @staticmethod
    def reshape_data(data):
        """Assumes that the data is 4 dimensional"""
        data_copy = data[:]
        data_copy = data_copy.transpose(2, 3)
        data_copy = data_copy.reshape(
            data_copy.shape[0] * data_copy.shape[1] * data_copy.shape[2],
            data_copy.shape[3],
        )
        return data_copy

    @staticmethod
    def unshape_data(data, shape):
        data_copy = data[:]
        data_copy = data_copy.reshape(shape[0], shape[1], shape[3], data.shape[1])
        return data_copy.transpose(2, 3)

    def forward(self, data):
        """Compute an RDM based on 1 - R^2 of the passed signals.

        :param data: A tensor of shape (classes/conditions, inputs, time)
        :return: 1 - Corr(network(class_1), network(class_2), ..., network(class_N))
        """
        reshaped_data = self.reshape_data(data)
        processed_data = self.unshape_data(self.process(reshaped_data), data.shape)

        return create_rdms(torch.squeeze(processed_data))


# for test
if __name__ == "__main__":
    net = RDM_MLP(2)
    net.eval()
    input = torch.as_tensor(
        np.array(
            [
                [[1, 2, 3, 4], [1, 2, 3, 4]],
                [[2, 3, 4, 5], [2, 3, 4, 5]],
                [[0, 1, 2, 3], [0, 1, 2, 3]],
            ],
            dtype="float32",
        )
    )
    # input = input.reshape(2, 4)
    print(f"{input.shape=}")
    # print(torch.corrcoef(input))
    print(net(input))
