import torch

from CONFIG import DTYPE_TORCH
from util.type_hints import *

import torch.nn as nn


class CnnEmbKin(nn.Module):
    def __init__(self, in_dim: int, emb_dim=1):
        super().__init__()

        self.in_dim = in_dim
        self.emb_dim = emb_dim
        kernel_size = 3

        self.encoder = nn.Sequential(
            nn.Conv1d(
                self.in_dim,
                20,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.Conv1d(
                20,
                20,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.Conv1d(
                20,
                20,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.Conv1d(
                20,
                10,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
        )

        self.mean_head = nn.Sequential(
            nn.Linear(10, self.emb_dim, dtype=DTYPE_TORCH),
        )

        self.var_head = nn.Sequential(
            nn.Linear(10, self.emb_dim, dtype=DTYPE_TORCH),
        )

    @staticmethod
    def reshape_data(data: torch.Tensor) -> torch.Tensor:
        """Reshapes Rnn output such that non-time dependent layers can easily process it.

        @param data: RNN output of hidden states.
        @return: Hidden states reshaped such that processing is as fast as possible.
        """
        data_copy = data[:]
        data_copy = data_copy.reshape(
            data.shape[0] * data.shape[1], data.shape[2], data.shape[3]
        )
        return data_copy

    def reshape_conv_data(
        self, data: torch.Tensor, new_shape: DataShape
    ) -> torch.Tensor:
        data_copy = data[:]
        data_copy = data_copy.reshape(
            new_shape[0], new_shape[1], data.shape[1], new_shape[3]
        )
        return data_copy.transpose(2, 3)

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute an RDM based on 1 - R^2 of the passed signals.

        @param data: A tensor of shape (participants x grasp phase x condition x inputs x time)
        @return: Embeddings and outputs.
        """

        def reparameterization(mean, var):
            epsilon = torch.rand(
                data.shape[0], data.shape[1], self.emb_dim, data.shape[3]
            )
            z = mean + var * epsilon
            return z

        reshaped_data = self.reshape_data(data)
        embeddings = self.reshape_conv_data(self.encoder(reshaped_data), data.shape)
        mean, var = self.mean_head(embeddings).transpose(2, 3), self.var_head(
            embeddings
        ).transpose(2, 3)
        reparams = reparameterization(mean, var)

        return mean, var, reparams
