import os.path

import torch
from networks.rdm_network import RdmMlp
from networks.autoencoder import Autoencoder, AutoEmbedder

from util.paths import get_subfiles


def get_last_network(network_path: str) -> str:
    """Searches for network with the highest epoch count and returns its full path.

    @param network_path: Folder path containing all saved network states.
    @return: Full path to the last trained network.
    """
    sub_files = get_subfiles(network_path)
    sub_files = list(
        filter(lambda x: len(x) == max(map(lambda x: len(x), sub_files)), sub_files)
    )
    network_name = sorted(sub_files, reverse=True)[0]
    return os.path.join(network_path, network_name)


def get_auto_inference_network(network_path: str, input_neurons: int) -> AutoEmbedder:
    """Loads an Autoencoder as embedder only.

    @param network_path: Path to the folder containing saved states of the autoencoder to be loaded.
    @param input_neurons: The number of input neurons the network has.
    @return: Network cut down to the embedder.
    """
    path = get_last_network(network_path)
    net = Autoencoder(input_neurons)
    _get_inference_network(path, net)
    embedder = AutoEmbedder(net)
    return embedder


def get_rdm_inference_network(network_path: str, input_neurons: int) -> RdmMlp:
    """Loads an RDM MLP.

    @param network_path: Path to the folder containing saved states of the RDM MLP to be loaded.
    @param input_neurons: The number of input neurons the network has.
    @return: RDM MLP.
    """
    path = get_last_network(network_path)
    net = RdmMlp(input_neurons)
    _get_inference_network(path, net)
    return net


def _get_inference_network(
    network_path: str, network_instance: torch.nn.Module
) -> None:
    """Loads a saved network into a passed instance and prepares it for evaluation/inference.

    @param network_path: The path to the network's saved state.
    @param network_instance: An initialized network instance with the same architecture as the saved model.
    @return: None, the model is loaded into the passed instance.
    """
    _get_network(network_path, network_instance)
    network_instance.eval()


def _get_network(network_path: str, network_instance: torch.nn.Module) -> None:
    """Loads a saved epoch state of a network into an instance of its architecture.

    @param network_path: The path to the network's saved state.
    @param network_instance: An initialized network instance with the same architecture as the saved model.
    @return: None, the model is loaded into the passed instance.
    """
    network_instance.load_state_dict(torch.load(network_path)["model_state_dict"])
