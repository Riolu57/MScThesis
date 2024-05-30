import torch

from CONFIG import DTYPE_TORCH
from util.type_hints import *

import torch.nn as nn


class CnnEmbKin(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(
                self.in_dim,
                20,
                kernel_size=3,
                stride=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.Conv1d(
                20,
                20,
                kernel_size=3,
                stride=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.Conv1d(
                20,
                20,
                kernel_size=3,
                stride=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.Conv1d(
                20,
                10,
                kernel_size=3,
                stride=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.Conv1d(
                10,
                1,
                kernel_size=3,
                stride=1,
                padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                1,
                10,
                kernel_size=3,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                10,
                20,
                kernel_size=3,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                20,
                20,
                kernel_size=3,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                20,
                20,
                kernel_size=3,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                20,
                self.out_dim,
                kernel_size=3,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
                padding_mode="zeros",
                dtype=DTYPE_TORCH,
            ),
            nn.ReLU(),
        )

    @staticmethod
    def reshape_data(data: DataConstruct) -> DataConstruct:
        """Reshapes data to be easily processed by network.
        @param data: A 4D DataConstruct, where dimension 2 are the input channels and 3 the time points.
        @return: A 2D DataConstruct reshaped, such that model(model.reshape_data(data)) is as fast as possible.
        """
        data_copy = data[:]
        data_copy = data_copy.reshape(
            data_copy.shape[0] * data_copy.shape[1] * data_copy.shape[2],
            data_copy.shape[3],
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
            shape[0], shape[1], shape[2], data.shape[1], shape[4]
        )
        return data_copy

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute an RDM based on 1 - R^2 of the passed signals.

        @param data: A tensor of shape (participants x grasp phase x condition x inputs x time)
        @return: Embeddings and outputs.
        """
        reshaped_data = self.reshape_data(data)
        embeddings = self.encoder(reshaped_data)
        outputs = self.unshape_data(self.decoder(embeddings), data.shape)

        return embeddings, outputs
