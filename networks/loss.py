import torch


def fro_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the squared sum of differences between the two passed tensors.
    @param output: A network's output.
    @param target: A target output. Needs to match size with "output" parameter.
    @return: A loss value.
    """
    return torch.sum((output - target).flatten() ** 2)


def pre_train_loss(
    mean: torch.Tensor,
    var: torch.Tensor,
    network_rdms: torch.Tensor,
    target_rdms: torch.Tensor,
) -> torch.Tensor:
    """Computes the squared sum of differences between embeddings and MSE between outputs.
    @param output: A network's embeddings and outputs.
    @param target: A target embedding and outputs. Needs to match size with "output" parameter.
    @return: A loss value.
    """
    KL_loss = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
    return KL_loss


def mixed_loss(
    mean: torch.Tensor,
    var: torch.Tensor,
    network_rdms: torch.Tensor,
    target_rdms: torch.Tensor,
) -> torch.Tensor:
    """Computes the squared sum of differences between embeddings and MSE between outputs. Further pushes embeddings
    towards a normal distribution.
    @param output: A network's embeddings and outputs.
    @param target: A target embedding and outputs. Needs to match size with "output" parameter.
    @return: A loss value.
    """
    fro_loss = torch.sum((network_rdms - target_rdms).flatten() ** 2)
    KL_loss = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
    return fro_loss + KL_loss
