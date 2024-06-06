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
    network_output: torch.Tensor,
    target_rdms: torch.Tensor,
    target_outputs: torch.Tensor,
) -> torch.Tensor:
    """Computes the squared sum of differences between embeddings and MSE between outputs.
    @param output: A network's embeddings and outputs.
    @param target: A target embedding and outputs. Needs to match size with "output" parameter.
    @return: A loss value.
    """
    MSE_loss = torch.nn.MSELoss()(network_output, target_outputs)
    return MSE_loss


def loss_wrapper_train(alpha: float):
    def mixed_loss(
        mean: torch.Tensor,
        var: torch.Tensor,
        network_rdms: torch.Tensor,
        network_output: torch.Tensor,
        target_rdms: torch.Tensor,
        target_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the squared sum of differences between embeddings and MSE between outputs. Further pushes embeddings
        towards a normal distribution.
        @param output: A network's embeddings and outputs.
        @param target: A target embedding and outputs. Needs to match size with "output" parameter.
        @return: A loss value.
        """
        fro_loss = alpha * torch.sum((network_rdms - target_rdms).flatten() ** 2)
        KL_loss = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
        MSE_loss = (1 - alpha) * torch.nn.MSELoss()(network_output, target_outputs)
        return fro_loss + KL_loss + MSE_loss

    return mixed_loss


def empty_loss(*args, **kwargs):
    raise NotImplementedError(
        "This should never have been used. An unreachable loss was used."
    )
