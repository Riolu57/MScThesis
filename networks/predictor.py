import torch

from CONFIG import DTYPE_TORCH
from util.type_hints import *

import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.decoder = nn.Sequential(
            nn.Linear(1, 10, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(10, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, 20, dtype=DTYPE_TORCH),
            nn.ReLU(),
            nn.Linear(20, self.out_dim, dtype=DTYPE_TORCH),
            nn.ReLU(),
        )
