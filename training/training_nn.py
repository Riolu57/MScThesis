from torch.nn import MSELoss

from networks.autoencoder import AUTOENCODER
from networks.rdm_network import RDM_MLP

from training.basic_loops import train_loop, test_loop
from networks.loss import fro_loss

from util.data import (
    prepare_rdm_data,
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
    val_loader = DataLoader(datasets[0], batch_size=3, shuffle=True)
    test_loader = DataLoader(datasets[0], batch_size=3, shuffle=True)

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
