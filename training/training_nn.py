# TypeHint only imports
from util.type_hints import *
from typing import Callable, Iterable
from torch.utils.data import Dataset

from torch.nn import MSELoss, RNN

from networks.autoencoder import AUTOENCODER
from networks.rdm_network import RDM_MLP

from training.basic_loops import train_loop, test_loop
from networks.loss import fro_loss

from data.loading import load_kinematics_data, load_eeg_data, split_data
from data.datasets import AutoDataset, prepare_rdm_data, prepare_rdm_data_rnn

from torch.utils.data import DataLoader
import torch

import numpy as np
import random


def train_network(
    model: torch.nn.Module,
    loss: Callable,
    datasets: Iterable[Dataset, Dataset, Dataset],
    seed: int,
    model_path: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
) -> None:
    """Trains a given network and saves it in the passed location.
    @param model: The network to be trained.
    @param loss: The loss function to be used for the model.
    @param datasets: The torch datasets containing training, validation and testing data; passed as a tuple or list.
    @param seed: A seed to be used for the experiment.
    @param model_path: Path to save the model after each epoch.
    @param epochs: How many epochs the model will be trained for.
    @param learning_rate: The learning rate; used for ADAM optimizer.
    @param alpha: The regularization parameter. [0, \inf]. Higher means stronger regularization.
    @return: None.
    """
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
    seed: int,
    eeg_path: str,
    kin_path: str,
    model_path: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
) -> None:
    """Trains an MLP which embedds EEG data to resemble kinematics RDMs as closely as possible.

    @param seed: Seed for the experiment.
    @param eeg_path: Path where EEG data is saved.
    @param kin_path: Path where Kinematics data is saved.
    @param model_path: Path where the model should be saved.
    @param epochs: For how many epochs the model should be trained for.
    @param learning_rate: Learning rate of ADAM.
    @param alpha: The regularization parameter. [0, \inf]. Higher means stronger regularization.
    @return: None.
    """
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


def train_autoencoder_eeg(
    seed: int,
    eeg_path: str,
    model_path: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
) -> None:
    """Trains an MLP Autoencoder using EEG data.

    @param seed: Seed for the experiment.
    @param eeg_path: Path where EEG data is saved.
    @param model_path: Path where the model should be saved.
    @param epochs: For how many epochs the model should be trained for.
    @param learning_rate: Learning rate of ADAM.
    @param alpha: The regularization parameter. [0, \inf]. Higher means stronger regularization.
    @return: None.
    """
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


def train_autoencoder_kin(
    seed: int,
    kin_path: str,
    model_path: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
) -> None:
    """Trains an MLP Autoencoder using EEG data.

    @param seed: Seed for the experiment.
    @param kin_path: Path where Kinematics data is saved.
    @param model_path: Path where the model should be saved.
    @param epochs: For how many epochs the model should be trained for.
    @param learning_rate: Learning rate of ADAM.
    @param alpha: The regularization parameter. [0, \inf]. Higher means stronger regularization.
    @return: None.
    """
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


def train_rnn_rdm(
    seed: int,
    eeg_path: str,
    kin_path: str,
    model_path: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
) -> None:
    """Trains an RNN which embedds EEG data to resemble kinematics RDMs as closely as possible.

    @param seed: Seed for the experiment.
    @param eeg_path: Path where EEG data is saved.
    @param kin_path: Path where Kinematics data is saved.
    @param model_path: Path where the model should be saved.
    @param epochs: For how many epochs the model should be trained for.
    @param learning_rate: Learning rate of ADAM.
    @param alpha: The regularization parameter. [0, \inf]. Higher means stronger regularization.
    @return: None.
    """
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
        train_loop(
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
        test_loop(f"{model_path}/data.txt", val_loader, model, MSELoss())
