import argparse

from util.paths import get_subdirs
import os

from CONFIG import *

from networks.mlp_emb_kin import MlpEmbKin
from networks.cnn_emb_kin import CnnEmbKin
from networks.rnn_emb_kin import RnnEmbKin

from torch.nn import MSELoss
from networks.loss import corr_loss, mixed_corr_loss

from training.training_nn import train_network_predictor

import torch
import random
import numpy as np

from data.loading import load_all_data

mlp_emb_kin_path = "./models/mlp_emb_kin/001"
rnn_emb_kin_path = "./models/rnn_emb_kin/001"
cnn_emb_kin_path = "./models/cnn_emb_kin/001"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type", action="store", choices=["MLP", "CNN", "RNN"], required=True
)
parser.add_argument(
    "--rdm_trained", action="store", choices=[True, False], required=True, type=bool
)
parser.add_argument(
    "--emb_dim",
    action="store",
    choices=[int(x) for x in get_subdirs(os.path.join(mlp_emb_kin_path, "embedder"))],
    required=True,
    type=int,
)
parser.add_argument(
    "--loss", action="store", choices=["MSE", "Corr", "Mixed"], required=True
)

args = parser.parse_args()

if args.model_type == "MLP":
    model = MlpEmbKin(16, args.emb_dim)
    model_path = mlp_emb_kin_path

elif args.model_type == "CNN":
    model = CnnEmbKin(16, args.emb_dim)
    model_path = cnn_emb_kin_path

elif args.model_type == "RNN":
    model = RnnEmbKin(16, args.emb_dim)
    model_path = rnn_emb_kin_path

if args.loss == "MSE":
    loss = MSELoss()

elif args.loss == "Corr":
    loss = corr_loss

elif args.loss == "Mixed":
    loss = mixed_corr_loss

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

eeg_data, kin_data = load_all_data(EEG_DATA_PATH, KIN_DATA_PATH)

train_network_predictor(
    model,
    eeg_data,
    kin_data,
    model_path,
    EPOCHS,
    PREDICTOR_LEARNING_RATE_SCHEDULE,
    ALPHA_PREDICTOR,
    args.emb_dim,
    loss_function=loss,
    pre_trained_only=args.rdm_trained,
)
