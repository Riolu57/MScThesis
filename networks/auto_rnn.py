from util.type_hints import *

import torch.nn as nn
import torch


class AutoRnn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        hidden_size = 1

        self.rnn = nn.RNN(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.out_layer = nn.Sequential(nn.Linear(hidden_size, self.out_dim), nn.ReLU())

    @staticmethod
    def preshape_data(data: DataConstruct) -> DataConstruct:
        """Reshapes data to be easily processed by the rnn part of the network.
        @param data: A 5D DataConstruct, where dimension 3 are the input channels and 4 the time points.
        @return: A 3D DataConstruct reshaped, such that self.rnn(self.preshape_data(data)) is as fast as possible.
        """
        data_copy = data[:]
        data_copy = data_copy.transpose(3, 4)
        data_copy = data_copy.reshape(
            data_copy.shape[0] * data_copy.shape[1] * data_copy.shape[2],
            data_copy.shape[3],
            data_copy.shape[4],
        )
        return data_copy

    @staticmethod
    def reshape_data(data: DataConstruct) -> DataConstruct:
        """Reshapes Rnn output such that non-time dependent layers can easily process it.

        @param data: RNN output of hidden states.
        @return: Hidden states reshaped such that processing is as fast as possible.
        """
        data_copy = data[:]
        data_copy = data_copy.reshape(
            data_copy.shape[0] * data_copy.shape[1], data_copy.shape[2]
        )

        return data_copy

    def unshape_data(self, data: DataConstruct, shape: DataShape) -> DataConstruct:
        """Reshapes model output to old shape, given the data.
        @param data: Model output data, to be reshaped.
        @param shape: The shape of the old data.
        @return: The model data reshaped to its original size.
        """
        data_copy = data[:]
        data_copy = data_copy.reshape(
            shape[0], shape[1], shape[2], shape[4], self.out_dim
        )
        return data_copy.transpose(3, 4)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Computes the model output given some data.
        @param data: Input data. Expected to be 5D, where dim 3 are the input channels.
        @return: Model output of nearly the same shape as input.
        """
        new_data = self.preshape_data(data)
        rnn_states, _ = self.rnn(new_data)
        linear_data = self.reshape_data(rnn_states)

        return self.unshape_data(self.out_layer(linear_data), data.shape)
