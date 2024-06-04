import torch


def fro_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the squared sum of differences between the two passed tensors.
    @param output: A network's output.
    @param target: A target output. Needs to match size with "output" parameter.
    @return: A loss value.
    """
    return torch.sum((output - target).flatten() ** 2)


def param_wrapper(alpha: float):
    def mixed_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the squared sum of differences between embeddings and MSE between outputs.
        @param output: A network's embeddings and outputs.
        @param target: A target embedding and outputs. Needs to match size with "output" parameter.
        @return: A loss value.
        """
        return alpha * torch.sum(
            (output[0] - target[0]).flatten() ** 2
        ) + (1 - alpha) * torch.nn.MSELoss()(output[1], target[2]) + torch.nn.KLDivLoss()(output[0].log(), target[1])

    return mixed_loss
