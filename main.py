import os.path

import torch
import numpy as np
import random

import sklearn

from CONFIG import (
    SEED,
    EEG_DATA_PATH,
    KIN_DATA_PATH,
    EPOCHS,
    LEARNING_RATE,
    ALPHA_EMBEDDER,
    ALPHA_PREDICTOR,
    PRE_TRAIN,
    PREDICTOR_LEARNING_RATE_SCHEDULE,
    EMB_DIM,
)
from training.training_nn import (
    train_eeg_emb,
    train_network_predictor,
    train_classical_predictor,
)

from training.training_classic import ana_pca, ana_ica

from data.loading import load_kinematics_data, load_eeg_data, load_all_data
from data.rdms import create_5D_rdms
from util.loss_vis import (
    plot_rdms,
    plot_loss_rdm,
)

from networks.mlp_emb_kin import MlpEmbKin
from networks.rnn_emb_kin import RnnEmbKin
from networks.cnn_emb_kin import CnnEmbKin

torch.backends.cudnn.deterministic = True


def train_embedders(seed, eeg_path, kin_path, epochs, pre_train, learning_rate, alpha):
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
            [MlpEmbKin(16, EMB_DIM), RnnEmbKin(16, EMB_DIM), CnnEmbKin(16, EMB_DIM)],
            [mlp_emb_kin_path, rnn_emb_kin_path, cnn_emb_kin_path],
        ):
            try:
                train_eeg_emb(
                    new_seed,
                    model,
                    eeg_data,
                    kin_data,
                    f"{model_path}/embedder/{EMB_DIM}/{new_seed}",
                    epochs,
                    pre_train,
                    learning_rate,
                    alpha,
                )

            except RuntimeError:
                with open(
                    f"{model_path}/embedder/{EMB_DIM}/{new_seed}/data.txt", "a+"
                ) as f:
                    f.write("COLLAPSED")

                continue


def train_kl_only_models(seed, eeg_path, kin_path, epochs, learning_rate, alpha):
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
            [MlpEmbKin(16, EMB_DIM), RnnEmbKin(16, EMB_DIM), CnnEmbKin(16, EMB_DIM)],
            [mlp_emb_kin_path, rnn_emb_kin_path, cnn_emb_kin_path],
        ):
            train_eeg_emb(
                new_seed,
                model,
                eeg_data,
                kin_data,
                f"{model_path}/embedder_kl/{EMB_DIM}/{new_seed}",
                epochs,
                epochs,
                learning_rate,
                alpha,
            )


def train_predictors(seed, eeg_path, kin_path, epochs, learning_rate, alpha):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.autograd.set_detect_anomaly(True)

    mlp_emb_kin_path = "./models/mlp_emb_kin/001"
    rnn_emb_kin_path = "./models/rnn_emb_kin/001"
    cnn_emb_kin_path = "./models/cnn_emb_kin/001"

    eeg_data, kin_data = load_all_data(eeg_path, kin_path)

    for model, model_path in zip(
        [MlpEmbKin(16, EMB_DIM), RnnEmbKin(16, EMB_DIM), CnnEmbKin(16, EMB_DIM)],
        [mlp_emb_kin_path, rnn_emb_kin_path, cnn_emb_kin_path],
    ):
        for pre_train in [True, False]:
            train_network_predictor(
                model,
                eeg_data,
                kin_data,
                f"{model_path}",
                epochs,
                learning_rate,
                alpha,
                pre_train,
            )


def train_classical_and_predictor(
    seed, eeg_path, kin_path, epoch, learning_rate, alpha
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.autograd.set_detect_anomaly(True)

    pca_predictor_path = "./models/pca/001"
    ica_predictor_path = "./models/ica/001"
    seed_path = "./models/cnn_emb_kin/001/embedder"

    eeg_data, kin_data = load_all_data(eeg_path, kin_path)

    for model, model_path in zip(
        [
            sklearn.decomposition.PCA(n_components=1),
            sklearn.decomposition.FastICA(n_components=1),
        ],
        [pca_predictor_path, ica_predictor_path],
    ):
        train_classical_predictor(
            model,
            eeg_data,
            kin_data,
            model_path,
            seed_path,
            epoch,
            learning_rate,
            alpha,
        )


# def plot_results(eeg_path, kin_path):
#     eeg_data = load_eeg_data(eeg_path)
#     kin_data = load_kinematics_data(kin_path)
#
#     kin_rdms = create_5D_rdms(kin_data)
#
#     training_loss_pca, pca_eeg, data = ana_pca(eeg_data, kin_rdms, eeg_data)
#     training_loss_ica, ica_eeg, data = ana_ica(eeg_data, kin_rdms, eeg_data)
#
#     eeg_rdms = [pca_eeg, ica_eeg]
#
#     plot_rdms(
#         eeg_rdms,
#         kin_rdms,
#         names=["PCA", "ICA", "RNN RDM"],
#     )
#
#     plot_loss("./training/models")


if __name__ == "__main__":
    train_embedders(
        SEED,
        EEG_DATA_PATH,
        KIN_DATA_PATH,
        EPOCHS,
        PRE_TRAIN,
        LEARNING_RATE,
        ALPHA_EMBEDDER,
    )
    # train_kl_only_models(
    #     SEED, EEG_DATA_PATH, KIN_DATA_PATH, EPOCHS, LEARNING_RATE, ALPHA_EMBEDDER
    # )
    # train_predictors(
    #     SEED,
    #     EEG_DATA_PATH,
    #     KIN_DATA_PATH,
    #     EPOCHS,
    #     PREDICTOR_LEARNING_RATE_SCHEDULE,
    #     ALPHA_PREDICTOR,
    # )
    # train_classical_and_predictor(
    #     SEED,
    #     EEG_DATA_PATH,
    #     KIN_DATA_PATH,
    #     EPOCHS,
    #     PREDICTOR_LEARNING_RATE_SCHEDULE,
    #     ALPHA_PREDICTOR,
    # )
