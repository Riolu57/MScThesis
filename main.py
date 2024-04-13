import torch
import numpy as np
import random

from CONFIG import SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA
from training.training_nn import (
    train_autoencoder_eeg,
    train_autoencoder_kin,
    train_rsa_embedding,
)

from training.training_classic import ana_pca

from util.data import load_eeg_data, load_kinematics_data
from util.loss_vis import plot_rdms, plot_loss

# TODO: ENSURE THAT CONCATENATIONS LEAVE ORDER OF MATRICES/CONDITIONS IN-TACT


def main(seed, eeg_path, kin_path, epochs, learning_rate, alpha):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.autograd.set_detect_anomaly(True)
    # eeg_data = load_eeg_data(eeg_path)
    # kin_data = load_kinematics_data(kin_path)

    # train_autoencoder_kin(
    #     seed,
    #     kin_path,
    #     "U:/Year 5/Thesis/training/models/kin_auto/001",
    #     epochs,
    #     learning_rate,
    #     alpha,
    # )
    #
    # train_autoencoder_eeg(
    #     seed,
    #     eeg_path,
    #     "U:/Year 5/Thesis/training/models/eeg_auto/001",
    #     epochs,
    #     learning_rate,
    #     alpha,
    # )

    train_rsa_embedding(
        seed,
        eeg_path,
        kin_path,
        "U:/Year 5/Thesis/training/models/rsa_emb/010",
        epochs,
        learning_rate,
        alpha,
    )

    # training_loss_pca, pca_eeg, pca_kin = ana_pca(eeg_data, kin_data)
    # plot_rdms(pca_eeg, pca_kin, names=["PCA"])

    plot_loss("U:/Year 5/Thesis/training/models")


if __name__ == "__main__":
    main(SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA)
