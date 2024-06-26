import os.path

import torch
import numpy as np
import random

from CONFIG import (
    SEED,
    EEG_DATA_PATH,
    KIN_DATA_PATH,
    EPOCHS,
    LEARNING_RATE,
    ALPHA,
    PRE_TRAIN,
)
from training.training_nn import train_eeg_emb

from training.training_classic import ana_pca, ana_ica

from data.loading import load_kinematics_data, load_eeg_data, load_all_data
from data.rdms import create_5D_rdms
from util.loss_vis import (
    plot_rdms,
    plot_loss,
    compute_rnn_rdms,
)

from networks.mlp_emb_kin import MlpEmbKin
from networks.rnn_emb_kin import RnnEmbKin
from networks.cnn_emb_kin import CnnEmbKin

torch.backends.cudnn.deterministic = True


def train_models(seed, eeg_path, kin_path, epochs, pre_train, learning_rate, alpha):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gen = np.random.default_rng(seed)

    torch.autograd.set_detect_anomaly(True)

    mlp_emb_kin_path = "./models/mlp_emb_kin/001"
    rnn_emb_kin_path = "./models/rnn_emb_kin/001"
    cnn_emb_kin_path = "./models/cnn_emb_kin/001"

    eeg_data, kin_data = load_all_data(eeg_path, kin_path)

    for new_seed in gen.integers(0, int("9" * (len(str(seed)))), 5):
        new_seed = int(new_seed)
        for model, model_path in zip(
            [MlpEmbKin(16), RnnEmbKin(16), CnnEmbKin(16)],
            [mlp_emb_kin_path, rnn_emb_kin_path, cnn_emb_kin_path],
        ):
            try:
                train_eeg_emb(
                    new_seed,
                    model,
                    eeg_data,
                    kin_data,
                    f"{model_path}/embedder/{new_seed}",
                    epochs,
                    pre_train,
                    learning_rate,
                    alpha,
                )

            except RuntimeError:
                with open(f"{model_path}/embedder/{new_seed}/data.txt", "a+") as f:
                    f.write("COLLAPSED")

                continue


def plot_results(eeg_path, kin_path):
    eeg_data = load_eeg_data(eeg_path)
    kin_data = load_kinematics_data(kin_path)

    kin_rdms = create_5D_rdms(kin_data)

    rnn_rdm_path = "training/models/rnn_rdm/002"

    training_loss_pca, pca_eeg = ana_pca(eeg_data, kin_rdms)
    training_loss_ica, ica_eeg = ana_ica(eeg_data, kin_rdms)
    rnn_rdms = compute_rnn_rdms(rnn_rdm_path, eeg_data)

    eeg_rdms = [pca_eeg, ica_eeg, rnn_rdms]

    plot_rdms(
        eeg_rdms,
        kin_rdms,
        names=["PCA", "ICA", "RNN RDM"],
    )

    plot_loss("./training/models")


def main(seed, eeg_path, kin_path, epochs, pre_train, learning_rate, alpha):
    train_models(seed, eeg_path, kin_path, epochs, pre_train, learning_rate, alpha)


if __name__ == "__main__":
    main(SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, PRE_TRAIN, LEARNING_RATE, ALPHA)
