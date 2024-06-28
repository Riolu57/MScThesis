import os.path

import torch


def get_best_network(network_path: str) -> str:
    """Searches for network with the highest epoch count and returns its full path.

    @param network_path: Folder path containing all saved network states.
    @return: Full path to the last trained network.
    """
    return os.path.join(network_path, "lowest_val_loss")


def get_rnn_rdm_network(network_path: str, input_neurons: int) -> torch.nn.RNN:
    """Loads an RDM RNN.

    @param network_path: Path to the folder containing saved states of the RDM MLP to be loaded.
    @param input_neurons: The number of input neurons the network has.
    @return: RDM RNN.
    """
    path = get_best_network(network_path)
    net = torch.nn.RNN(input_neurons, 1)
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
