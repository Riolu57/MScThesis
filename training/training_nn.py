from torch.nn import MSELoss, RNN

from networks.autoencoder import AUTOENCODER
from networks.rdm_network import RDM_MLP

from training.basic_loops import train_loop, test_loop, rnn_train_loop, rnn_test_loop
from networks.loss import fro_loss

from util.data import (
    prepare_rdm_data,
    prepare_rdm_data_rnn,
    split_data,
    AutoDataset,
    load_eeg_data,
    load_kinematics_data,
)

from torch.utils.data import DataLoader
import torch

import numpy as np
import random


def train_network(
    model, loss, datasets, seed, model_path, epochs, learning_rate, alpha
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_loader = DataLoader(datasets[0], batch_size=10, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=3, shuffle=True)
    test_loader = DataLoader(datasets[2], batch_size=3, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(
            f"======================================== CURRENT EPOCH: {epoch + 1} ========================================"
        )
        train_loop(
            f"{model_path}/data.txt",
            train_loader,
            model,
            loss,
            optimizer,
            alpha,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            f"{model_path}/epoch_{epoch}",
        )
        test_loop(f"{model_path}/data.txt", val_loader, model, loss)


def train_rsa_embedding(
    seed, eeg_path, kin_path, model_path, epochs, learning_rate, alpha
):
    model = RDM_MLP(16)
    train_data, val_data, test_data = prepare_rdm_data(eeg_path, kin_path)

    train_network(
        model,
        fro_loss,
        [train_data, val_data, test_data],
        seed,
        model_path,
        epochs,
        learning_rate,
        alpha,
    )


def train_autoencoder_eeg(seed, eeg_path, model_path, epochs, learning_rate, alpha):
    train_data, val_data, test_data = split_data(load_eeg_data(eeg_path))

    train_data = AutoDataset(train_data)
    val_data = AutoDataset(val_data)
    test_data = AutoDataset(test_data)

    model = AUTOENCODER(train_data[0][0].shape[2])

    print(f"Model initialized with {train_data[0][0].shape[2]} input neurons.")

    train_network(
        model,
        MSELoss(),
        [train_data, val_data, test_data],
        seed,
        model_path,
        epochs,
        learning_rate,
        alpha,
    )


def train_autoencoder_kin(seed, kin_path, model_path, epochs, learning_rate, alpha):
    train_data, val_data, test_data = split_data(load_kinematics_data(kin_path))

    train_data = AutoDataset(train_data)
    val_data = AutoDataset(val_data)
    test_data = AutoDataset(test_data)

    model = AUTOENCODER(train_data[0][0].shape[2])

    train_network(
        model,
        MSELoss(),
        [train_data, val_data, test_data],
        seed,
        model_path,
        epochs,
        learning_rate,
        alpha,
    )


def train_rnn_rdm(seed, eeg_path, kin_path, model_path, epochs, learning_rate, alpha):
    model = RNN(
        input_size=16,
        hidden_size=1,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    )
    train_data, val_data, test_data = prepare_rdm_data_rnn(eeg_path, kin_path)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=3, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(
            f"======================================== CURRENT EPOCH: {epoch + 1} ========================================"
        )
        rnn_train_loop(
            f"{model_path}/data.txt",
            train_loader,
            model,
            MSELoss(),
            optimizer,
            alpha,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            },
            f"{model_path}/epoch_{epoch}",
        )
        rnn_test_loop(f"{model_path}/data.txt", val_loader, model, MSELoss())


# Create RNN that takes EEG data and then predicts Kin Data, with a hidden layer of size 1
