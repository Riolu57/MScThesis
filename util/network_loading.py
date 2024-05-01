import os.path

import torch
import torch.nn as nn
from networks.rdm_network import RDM_MLP
from networks.autoencoder import AUTOENCODER

from util.data import create_eeg_data, create_rdms

from util.data import get_subfiles


def get_last_network(network_path):
    sub_files = get_subfiles(network_path)
    sub_files = list(
        filter(lambda x: len(x) == max(map(lambda x: len(x), sub_files)), sub_files)
    )
    network_name = sorted(sub_files, reverse=True)[0]
    return os.path.join(network_path, network_name)


def get_auto_inference_network(network_path, input_neurons):
    path = get_last_network(network_path)
    net = AUTOENCODER(input_neurons)
    _get_inference_network(path, net)
    embedder = AutoEmbedder(net)
    return embedder


def get_rdm_inference_network(network_path, input_neurons):
    path = get_last_network(network_path)
    net = RDM_MLP(input_neurons)
    _get_inference_network(path, net)
    return net


def _get_inference_network(network_path, network_instance):
    _get_network(network_path, network_instance)
    network_instance.eval()


def _get_network(network_path, network_instance):
    network_instance.load_state_dict(torch.load(network_path)["model_state_dict"])


class AutoEmbedder(nn.Module):
    def __init__(self, architecture):
        super().__init__()

        subnetwork = RDM_MLP(architecture.process[0].in_features)
        encoder = []

        for idx in range(len(subnetwork.process)):
            encoder.append(architecture.process[idx])

        self.process = nn.Sequential(*encoder)

    def forward(self, data):
        tensor_data = self.reshape_data(torch.as_tensor(data))
        processed_data = self.process(tensor_data)
        unshaped_data = torch.squeeze(self.unshape_data(processed_data, data.shape))
        rdm_ready_data = unshaped_data.reshape(
            unshaped_data.shape[0] * unshaped_data.shape[1],
            unshaped_data.shape[2],
            unshaped_data.shape[3],
        )

        return create_rdms(rdm_ready_data)

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
        data_copy = data_copy.reshape(
            shape[0], shape[1], shape[2], shape[4], data.shape[1]
        )
        return data_copy.transpose(3, 4)
