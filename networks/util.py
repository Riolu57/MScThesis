import torch
from networks.cnn_emb_kin import CnnEmbKin

from data.rdms import create_5D_rdms
from data.reshaping import rnn_reshaping, rnn_unshaping, cnn_unshaping

from typing import Tuple


def add_noise(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.data += torch.rand(param.shape, generator=torch.Generator()) * torch.max(
            param.data
        )


def get_rdms(
    data: torch.Tensor, model: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings, outputs = model(data)
    if isinstance(model, CnnEmbKin):
        rdms = create_5D_rdms(cnn_unshaping(embeddings, data.shape))
    else:
        rdms = create_5D_rdms(rnn_unshaping(embeddings, data.shape))

    return rdms, outputs
