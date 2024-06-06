import torch
from networks.cnn_emb_kin import CnnEmbKin

from data.rdms import create_5D_rdms
from data.reshaping import rnn_reshaping, rnn_unshaping, cnn_unshaping

from typing import Tuple


def get_rdms(
    data: torch.Tensor, model: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, var, embeddings, outputs = model(data)
    if isinstance(model, CnnEmbKin):
        rdms = create_5D_rdms(cnn_unshaping(embeddings, data.shape))
    else:
        rdms = create_5D_rdms(rnn_unshaping(embeddings, data.shape))

    return mean, var, rdms, outputs
