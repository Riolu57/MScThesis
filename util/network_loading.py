import os.path

import torch
import torch.nn as nn
from networks.rdm_network import RDM_MLP
from networks.autoencoder import AUTOENCODER

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
    def __int__(self, architecture):
        super().__init__()

        self.subnetwork = RDM_MLP(architecture.process[0].in_features)
        encoder = []

        for idx in range(len(self.subnetwork.process)):
            encoder.append(architecture.process[idx])

        self.process = nn.Sequential(*encoder)

    def forward(self, data):
        modded_data = self.subnetwork.reshape_data(data)
        processed_data = self.process(modded_data)
        return self.subnetwork.unshape_data(processed_data, data.shape)
