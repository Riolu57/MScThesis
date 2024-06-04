import os.path

import torch
import numpy as np
import random

from CONFIG import SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA
from training.training_nn import (
    train_autoencoder_eeg,
    train_autoencoder_kin,
    train_rsa_embedding,
    train_eeg_emb_kin,
)

from training.training_classic import ana_pca, ana_ica

from data.loading import load_kinematics_data, load_eeg_data
from data.rdms import create_5D_rdms
from util.loss_vis import (
    plot_rdms,
    plot_loss,
    compute_rdm_rdms,
    compute_auto_rdms,
    compute_rnn_rdms,
)

from networks.mlp_emb_kin import MlpEmbKin
from networks.rnn_emb_kin import RnnEmbKin
from networks.cnn_emb_kin import CnnEmbKin


def train_models(seed, eeg_path, kin_path, epochs, learning_rate, gamma):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gen = np.random.default_rng(seed)

    torch.autograd.set_detect_anomaly(True)

    kin_auto_path = "./models/kin_auto/001"
    eeg_auto_path = "./models/eeg_auto/001"
    mlp_emb_kin_path = "./models/mlp_emb_kin/001"
    rnn_emb_kin_path = "./models/rnn_emb_kin/001"
    cnn_emb_kin_path = "./models/cnn_emb_kin/001"

    for new_seed in gen.integers(0, int("9" * (len(str(seed)))), 5):
        for alpha in range(0, 11, 2):

            new_seed = int(new_seed)

            try:

                train_eeg_emb_kin(
                    new_seed,
                    MlpEmbKin(16, 19),
                    eeg_path,
                    kin_path,
                    f"{mlp_emb_kin_path}/a_{alpha}/{new_seed}",
                    epochs,
                    learning_rate,
                    alpha / 10,
                    gamma,
                )

            except RuntimeError:
                with open(f"{mlp_emb_kin_path}/a_{alpha}/{new_seed}/data.txt", "a+") as f:
                    f.write("COLLAPSED")

                continue



            # train_eeg_emb_kin(
            #     new_seed,
            #     RnnEmbKin(16, 19),
            #     eeg_path,
            #     kin_path,
            #     f"{rnn_emb_kin_path}/a_{alpha}/{new_seed}",
            #     epochs,
            #     learning_rate,
            #     alpha / 10,
            #     gamma,
            # )

            # train_eeg_emb_kin(
            #     new_seed,
            #     CnnEmbKin(16, 19),
            #     eeg_path,
            #     kin_path,
            #     f"{cnn_emb_kin_path}/a_{alpha}/{new_seed}",
            #     epochs,
            #     learning_rate,
            #     alpha / 10,
            #     gamma,
            # )


def plot_results(eeg_path, kin_path):
    eeg_data = load_eeg_data(eeg_path)
    kin_data = load_kinematics_data(kin_path)

    kin_rdms = create_5D_rdms(kin_data)

    kin_auto_path = "./training/models/kin_auto/002"
    eeg_auto_path = "./training/models/eeg_auto/001"
    rnn_rdm_path = "training/models/rnn_rdm/002"

    training_loss_pca, pca_eeg = ana_pca(eeg_data, kin_rdms)
    training_loss_ica, ica_eeg = ana_ica(eeg_data, kin_rdms)
    kin_auto_rdms = compute_auto_rdms(kin_auto_path, kin_data)
    eeg_auto_rdms = compute_auto_rdms(eeg_auto_path, eeg_data)
    rdm_rdms = compute_rdm_rdms(rsa_path, eeg_data)
    rnn_rdms = compute_rnn_rdms(rnn_rdm_path, eeg_data)

    eeg_rdms = [pca_eeg, ica_eeg, kin_auto_rdms, eeg_auto_rdms, rdm_rdms, rnn_rdms]

    plot_rdms(
        eeg_rdms,
        kin_rdms,
        names=["PCA", "ICA", "Kin Auto", "EEG Auto", "RDM Emb.", "RNN RDM"],
    )

    plot_loss("./training/models")


def main(seed, eeg_path, kin_path, epochs, learning_rate, alpha):
    train_models(seed, eeg_path, kin_path, epochs, learning_rate, alpha)


if __name__ == "__main__":
    main(SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA)
