import torch


def fro_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the squared sum of differences between the two passed tensors.
    @param output: A network's output.
    @param target: A target output. Needs to match size with "output" parameter.
    @return: A loss value.
    """
    return torch.sum((output - target).flatten() ** 2)
