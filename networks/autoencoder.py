from util.type_hints import *

import torch.nn as nn
import torch

from networks.rdm_network import RDM_MLP


class AUTOENCODER(nn.Module):
    """Participants need to be dim 0, grasp phase dim 1, Conditions need to be dim 2, number of channels dim 3, data/time points dim 4"""

    def __init__(self, in_dim: int):
        super().__init__()

        self.in_dim = in_dim

        ### Compex list manipulation to create reversed copy of sequential model of rdm_network ###
        # Create RDM MLP to get architecture
        subnetwork = RDM_MLP(self.in_dim)

        # Get layers and copy for later modifications
        layers = subnetwork.process
        encoding = [layer for layer in layers]

        # Get copy for later modification
        decoding_start = encoding[:]
        # Get non ReLU layers and inverse order
        decoding_start = decoding_start[::2][::-1]
        # Swap input and output dimensions
        decoder_layers = list(map(self.create_swapped_linear, decoding_start))
        # Create decoder
        decoding = [
            layer
            for combination in zip(decoder_layers, [nn.ReLU() for i in decoder_layers])
            for layer in combination
        ]

        # Combine encoder and decoder and assign to expected variable
        encoding.extend(decoding)
        self.process = nn.Sequential(*encoding)

    @staticmethod
    def create_swapped_linear(layer: torch.nn.Linear) -> torch.nn.Linear:
        """Creates a linear layer with swapped in-/output feature dimensions that is returned.
        @param layer: Linear layer to be swapped.
        @return: Copy of passed layer where input/output dimensions are swapped.
        """
        return nn.Linear(layer.out_features, layer.in_features)

    @staticmethod
    def reshape_data(data: DataConstruct) -> DataConstruct:
        """Reshapes data to be easily processed by network.
        @param data: A 5D DataConstruct, where dimension 3 are the input channels and 4 the time points.
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
        data_copy = data_copy.reshape(shape[0], shape[1], shape[2], shape[4], shape[3])
        return data_copy.transpose(3, 4)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Computes the model output given some data.
        @param data: Input data. Expected to be 5D, where dim 3 are the input channels.
        @return: Model output of same shape as input.
        """
        new_data = self.reshape_data(data)

        return self.unshape_data(self.process(new_data), data.shape)
