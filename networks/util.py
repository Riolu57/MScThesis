from typing import Tuple

import torch

from data.rdms import create_rdms


def get_rdms(
    data: torch.Tensor, model: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, var, embeddings = model(data)
    rdms = create_rdms(torch.squeeze(embeddings))
    return mean, var, rdms


def save_model(
    model: torch.nn.Module,
    model_path: str,
    optimizer: torch.optim.Optimizer,
    epoch: int,
):
    """Saves a model under the passed path.

    @param model: The Network to be saved.
    @param model_path: The path where the model should be saved.
    @param optimizer: The optimizer parameters, such that training could be continued.
    @param epoch: The current epoch.
    @return: None.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        f"{model_path}",
    )
