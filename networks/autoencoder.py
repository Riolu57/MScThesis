import torch.nn as nn
import torch

from CONFIG import DTYPE_TORCH
from networks.rdm_network import RDM_MLP


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
        decoder_layers = list(map(self.swap_dims, decoding_start))
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
    def swap_dims(layer):
        cur_layer = eval("nn." + repr(layer))
        cur_layer.in_features = layer.out_features
        cur_layer.out_features = layer.in_features
        return cur_layer

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
