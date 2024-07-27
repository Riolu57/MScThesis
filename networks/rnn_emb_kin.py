from util.type_hints import *

from CONFIG import DTYPE_TORCH

import torch.nn as nn
import torch


class RnnEmbKin(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 1):
        super().__init__()

        self.in_dim = in_dim
        self.emb_dim = emb_dim

        self.hidden_size = 10

        self.rnn = nn.RNN(
            input_size=self.in_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.mean_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.emb_dim, dtype=DTYPE_TORCH),
        )

        self.var_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.emb_dim, dtype=DTYPE_TORCH),
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
        return data_copy.transpose(1, 2)

    def unshape_data(
        self, rnn_states: torch.Tensor, new_shape: DataShape
    ) -> torch.Tensor:
        """Reshapes RNN states to 4D data again.

        @param rnn_states: RNN states to be reshaped.
        @param new_shape: New shape of RNN data.
        @return: 4D Data
        """
        data_copy = rnn_states[:]
        return data_copy.reshape(
            new_shape[0], new_shape[1], new_shape[3], self.hidden_size
        )

    def forward(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the model output given some data.
        @param data: Input data. Expected to be 5D, where dim 3 are the input channels.
        @return: Model output of nearly the same shape as input.
        """

        def reparameterization(mean, var):
            epsilon = torch.rand(
                data.shape[0], data.shape[1], self.emb_dim, data.shape[3]
            )
            z = mean + var * epsilon
            return z

        reshaped_data = self.reshape_data(data)
        rnn_states = self.unshape_data(self.rnn(reshaped_data)[0], data.shape)
        mean, var = self.mean_head(rnn_states).transpose(2, 3), self.var_head(
            rnn_states
        ).transpose(2, 3)
        reparams = reparameterization(mean, var)

        return mean, var, reparams
