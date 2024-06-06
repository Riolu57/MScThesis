# TypeHint only imports
import os.path

from util.type_hints import *
from typing import Callable, Iterable
from torch.utils.data import Dataset
from torch.nn import MSELoss

from training.basic_loops import nn_train_loop, nn_test_loop
from networks.loss import loss_wrapper_train, pre_train_loss, empty_loss
from networks.autoencoder import Autoencoder

from data.datasets import prepare_eeg_emb_kin_data, split_data, AutoDataset

from data.loading import load_eeg_data, load_kinematics_data

from torch.utils.data import DataLoader
import torch

import numpy as np
import random


def train_network(
    model: torch.nn.Module,
    loss: Callable,
    datasets: Iterable[Dataset],
    seed: int,
    model_path: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
    pre_train: int,
    pre_train_loss_func: Callable,
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
    @param pre_train: The number of epochs to only train the output and not embedding.
    @param pre_train_loss: The pre-train loss function.
    @return: None.
    """
    # Make sure that files and folder exist properly
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        open(os.path.join(model_path, "data.txt"), "a").close()

    elif not os.path.exists(os.path.join(model_path, "data.txt")):
        open(os.path.join(model_path, "data.txt"), "a").close()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_loader = DataLoader(datasets[0], batch_size=10, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=3, shuffle=True)
    test_loader = DataLoader(datasets[2], batch_size=3, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lowest_loss = np.infty

    for epoch in range(epochs):
        print(
            f"======================================== CURRENT EPOCH: {epoch + 1} ========================================"
        )

        if epoch < pre_train:
            nn_train_loop(
                f"{model_path}/data.txt",
                train_loader,
                model,
                pre_train_loss_func,
                optimizer,
                alpha,
            )

            test_loss = nn_test_loop(
                f"{model_path}/data.txt", val_loader, model, pre_train_loss
            )

        else:
            nn_train_loop(
                f"{model_path}/data.txt",
                train_loader,
                model,
                loss,
                optimizer,
                alpha,
            )

            test_loss = nn_test_loop(f"{model_path}/data.txt", val_loader, model, loss)

            if test_loss < lowest_loss:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    f"{model_path}/lowest_val_loss",
                )
                lowest_loss = test_loss

        if test_loss < lowest_loss:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                f"{model_path}/lowest_val_loss",
            )
            lowest_loss = test_loss


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

    model = Autoencoder(train_data[0][0].shape[2])

    train_network(
        model,
        MSELoss(),
        [train_data, val_data, test_data],
        seed,
        model_path,
        epochs,
        learning_rate,
        alpha,
        0,
        empty_loss,
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

    model = Autoencoder(train_data[0][0].shape[2])

    train_network(
        model,
        MSELoss(),
        [train_data, val_data, test_data],
        seed,
        model_path,
        epochs,
        learning_rate,
        alpha,
        0,
        empty_loss,
    )


def train_eeg_emb_kin(
    seed: int,
    model: torch.nn.Module,
    eeg_path: str,
    kin_path: str,
    model_path: str,
    epochs: int,
    learning_rate: float,
    alpha: float,
    gamma: float,
) -> None:
    """Trains an RNN which embedds EEG data to resemble kinematics RDMs as closely as possible.

    @param seed: Seed for the experiment.
    @param model: The model which will be trained.
    @param eeg_path: Path where EEG data is saved.
    @param kin_path: Path where Kinematics data is saved.
    @param model_path: Path where the model should be saved.
    @param epochs: For how many epochs the model should be trained for.
    @param learning_rate: Learning rate of ADAM.
    @param alpha: Fro loss between embedding and Kin RDMs. [0, 1]. Also regulates MSE loss between prediction and kin (1 - alpha).
    @param gamma: The regularization parameter. [0, \inf]. Higher means stronger regularization.
    @return: None.
    """

    data = prepare_eeg_emb_kin_data(eeg_path, kin_path, seed)

    train_network(
        model,
        loss_wrapper_train(alpha),
        data,
        seed,
        model_path,
        epochs,
        learning_rate,
        gamma,
        50,
        pre_train_loss,
    )
