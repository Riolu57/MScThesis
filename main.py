import torch
import numpy as np
import random

from CONFIG import SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA
from training.training_nn import (
    train_autoencoder_eeg,
    train_autoencoder_kin,
    train_rsa_embedding,
    train_rnn_rdm,
    train_eeg_kin_rnn,
)

from training.training_classic import ana_pca, ana_ica

from data.loading import load_kinematics_data, load_eeg_data
from data.rdms import create_kin_rdms
from util.loss_vis import (
    plot_rdms,
    plot_loss,
    compute_rdm_rdms,
    compute_auto_rdms,
    compute_rnn_rdms,
)


def main(seed, eeg_path, kin_path, epochs, learning_rate, alpha):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.autograd.set_detect_anomaly(True)
    eeg_data = load_eeg_data(eeg_path)
    kin_data = load_kinematics_data(kin_path)

    kin_rdms = create_kin_rdms(kin_data)

    kin_auto_path = "./training/models/kin_auto/001"
    eeg_auto_path = "./training/models/eeg_auto/001"
    rsa_path = "./training/models/rsa_emb/010"
    rnn_rdm_path = "training/models/rnn_rdm/001"
    rnn_auto_path = "training/models/rnn_auto/001"

    # train_autoencoder_kin(
    #     seed,
    #     kin_path,
    #     kin_auto_path,
    #     epochs,
    #     learning_rate,
    #     alpha,
    # )
    #
    # train_autoencoder_eeg(
    #     seed,
    #     eeg_path,
    #     eeg_auto_path,
    #     epochs,
    #     learning_rate,
    #     alpha,
    # )

    # train_rsa_embedding(
    #     seed,
    #     eeg_path,
    #     kin_path,
    #     rsa_path,
    #     epochs,
    #     learning_rate,
    #     alpha,
    # )

    # train_rnn_rdm(
    #     seed,
    #     eeg_path,
    #     kin_path,
    #     rnn_rdm_path,
    #     epochs,
    #     learning_rate,
    #     alpha,
    # )

    # train_eeg_kin_rnn(
    #     seed, eeg_path, kin_path, rnn_auto_path, epochs, learning_rate, alpha
    # )

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


if __name__ == "__main__":
    main(SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA)
