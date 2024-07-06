import torch

from CONFIG import DTYPE_TORCH
from util.type_hints import *

import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

        self.out_dim = out_dim

        self.decoder = nn.Sequential(
            nn.Linear(1, 10, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(10, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, self.out_dim, dtype=DTYPE_TORCH),
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

        reshaped_data = self.reshape_data(data)
        prediction = self.decoder(reshaped_data)

        return prediction.transpose(2, 3)
