# TypeHint only imports
import os.path
import time

from util.type_hints import *
from util.prettier import pcolors
from typing import Callable, Iterable
from torch.utils.data import Dataset

from training.basic_loops import nn_train_loop, nn_test_loop
from networks.loss import mixed_loss, pre_train_loss
from networks.util import save_model

from data.datasets import prepare_eeg_rdm_data

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
    lowest_pre_loss = np.infty

    pre_train_times = []
    train_times = []

    for epoch in range(epochs):
        current_time = time.time()
        print(
            f"\n{pcolors.BOLD}============================ CURRENT EPOCH: {epoch + 1} ({model.__class__.__name__}) ============================{pcolors.ENDC}"
        )

        if epoch < pre_train:
            save_name = "lowest_val_loss_pre_train"
            cur_loss = pre_train_loss_func

        else:
            save_name = "lowest_val_loss"
            cur_loss = loss

        nn_train_loop(
            f"{model_path}/data.txt",
            train_loader,
            model,
            cur_loss,
            optimizer,
            alpha,
        )

        test_loss = nn_test_loop(f"{model_path}/data.txt", val_loader, model, cur_loss)

        if epoch < pre_train:
            if test_loss < lowest_pre_loss:
                save_model(model, f"{model_path}/{save_name}", optimizer, epoch)
                lowest_pre_loss = test_loss

            pre_train_times.append(current_time)
            if len(pre_train_times) > 1:
                current_est = np.mean(np.diff(pre_train_times))
                min, sec = divmod(current_est * (pre_train - epoch), 60)
                print(
                    f"{pcolors.BOLD}Time Estimate (Pre Train): {int(min)}m {int(sec)}s{pcolors.ENDC}"
                )

        elif epoch >= pre_train:
            if test_loss < lowest_loss:
                save_model(model, f"{model_path}/{save_name}", optimizer, epoch)
                lowest_loss = test_loss

            train_times.append(current_time)
            if len(train_times) > 1:
                current_est = np.mean(np.diff(train_times))
                min, sec = divmod(current_est * (epochs - epoch), 60)
                print(
                    f"{pcolors.BOLD}Time Estimate (Train): {int(min)}m {int(sec)}s{pcolors.ENDC}"
                )


def train_eeg_emb(
    seed: int,
    model: torch.nn.Module,
    eeg_data: DataConstruct,
    kin_data: DataConstruct,
    model_path: str,
    epochs: int,
    pre_train: int,
    learning_rate: float,
    alpha: float,
) -> None:
    """Trains an RNN which embedds EEG data to resemble kinematics RDMs as closely as possible.

    @param seed: Seed for the experiment.
    @param model: The model which will be trained.
    @param eeg_path: Path where EEG data is saved.
    @param kin_path: Path where Kinematics data is saved.
    @param model_path: Path where the model should be saved.
    @param epochs: For how many epochs the model should be trained for.
    @param learning_rate: Learning rate of ADAM.
    @param alpha: The regularization parameter. [0, \inf]. Higher means stronger regularization.
    @return: None.
    """

    data = prepare_eeg_rdm_data(eeg_data, kin_data)

    train_network(
        model,
        mixed_loss,
        data,
        seed,
        model_path,
        epochs,
        learning_rate,
        alpha,
        pre_train,
        pre_train_loss,
    )
