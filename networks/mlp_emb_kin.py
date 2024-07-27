import torch

from CONFIG import DTYPE_TORCH
from util.type_hints import *

import torch.nn as nn


class MlpEmbKin(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 1):
        super().__init__()

        self.in_dim = in_dim
        self.emb_dim = emb_dim

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
            nn.Linear(10, self.emb_dim, dtype=DTYPE_TORCH),
        )

        self.var_head = nn.Sequential(
            nn.Linear(10, self.emb_dim, dtype=DTYPE_TORCH),
        )

    @staticmethod
    def reshape_data(data: DataConstruct) -> DataConstruct:
        """Reshapes data to be easily processed by network.
        @param data: A 4D DataConstruct, where dimension 2 are the input channels and 3 the time points.
        @return: A 4D DataConstruct reshaped, such that model(model.reshape_data(data)) is as fast as possible.
        """
        data_copy = data[:]
        return data_copy.transpose(2, 3)

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute an RDM based on 1 - R^2 of the passed signals.

        @param data: A tensor of shape (participants * grasp phase, condition, inputs, time)
        @return: Embeddings and outputs.
        """

        def reparameterization(mean, var):
            epsilon = torch.rand(
                data.shape[0], data.shape[1], self.emb_dim, data.shape[3]
            )
            z = mean + var * epsilon
            return z

        reshaped_data = self.reshape_data(data)
        encodings = self.encoder(reshaped_data)
        mean, var = self.mean_head(encodings).transpose(2, 3), self.var_head(
            encodings
        ).transpose(2, 3)
        reparams = reparameterization(mean, var)

        return mean, var, reparams
