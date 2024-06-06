import torch

from CONFIG import DTYPE_TORCH
from util.type_hints import *

import torch.nn as nn

from data.rdms import create_5D_rdms


class MlpEmbKin(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 10, dtype=DTYPE_TORCH),
            nn.ReLU(),
        )

        self.mean_head = nn.Sequential(
            nn.Linear(10, 1, dtype=DTYPE_TORCH),
        )

        self.var_head = nn.Sequential(
            nn.Linear(10, 1, dtype=DTYPE_TORCH),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 10, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(10, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, self.out_dim, dtype=DTYPE_TORCH),
            nn.ReLU(),
        )

    @staticmethod
    def reshape_data(data: DataConstruct) -> DataConstruct:
        """Reshapes data to be easily processed by network.
        @param data: A 4D DataConstruct, where dimension 2 are the input channels and 3 the time points.
        @return: A 2D DataConstruct reshaped, such that model(model.reshape_data(data)) is as fast as possible.
        """
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
    def unshape_data(data: DataConstruct, shape: DataShape) -> DataConstruct:
        """Reshapes model output to old shape, given the data.
        @param data: Model output data, to be reshaped.
        @param shape: The shape of the old data.
        @return: The model data reshaped to its original size.
        """
        data_copy = data[:]
        data_copy = data_copy.reshape(
            shape[0], shape[1], shape[2], shape[4], data.shape[1]
        )
        return data_copy.transpose(3, 4)

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute an RDM based on 1 - R^2 of the passed signals.

        @param data: A tensor of shape (participants x grasp phase x condition x inputs x time)
        @return: Embeddings and outputs.
        """

        def reparameterization(mean, var):
            epsilon = torch.randn_like(var)
            z = mean + var * epsilon
            return z

        reshaped_data = self.reshape_data(data)
        encodings = self.encoder(reshaped_data)
        mean, var = self.mean_head(encodings), self.var_head(encodings)
        reparams = reparameterization(mean, var)
        outputs = self.unshape_data(self.decoder(reparams), data.shape)

        return mean, var, reparams, outputs
