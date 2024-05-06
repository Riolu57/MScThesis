import numpy as np

from util.type_hints import *
from typing import Callable
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator

import torch
from data.rdms import create_rdms
from data.reshaping import rnn_reshaping, rnn_unshaping

from networks import Autoencoder, RdmMlp, fro_loss


def train_loop(
    path: str,
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    alpha: float,
):
    """Trains a model for a single epoch. Supports Autoencoders, MLPs and RNNS.
    @param path: Path for the model performance (i.e. loss values) to be saved.
    @param dataloader: DataLoader containing the training data.
    @param model: The model to be trained.
    @param loss_fn: The loss function to be used. Must return a single number and only accept a prediction and target.
    @param optimizer: A torch optimizer to be used on the model.
    @param alpha: Regularization parameter for L2/squared weight regularization. Higher means stronger regularization.
    @return: None.
    """
    # Set model to training mode
    model.train()

    full_model_loss = 0
    full_reg_loss = 0

    # Go through training data according to batches
    for batch_idx, (X, y) in enumerate(dataloader):
        print(f"Current batch idx: {batch_idx}")

        # Reset gradients
        optimizer.zero_grad()

        # Compute prediction and loss based on architecture
        if isinstance(model, torch.nn.RNN):
            rnn_data = rnn_reshaping(X)
            states, _ = model(rnn_data)
            pred = create_rdms(torch.squeeze(rnn_unshaping(states, X.shape)))
        elif isinstance(model, (Autoencoder, RdmMlp)):
            pred = model(X)
        else:
            raise NotImplemented("This type of architecture is not yet supported.")

        loss = loss_fn(pred, y)

        # Compute regularization loss
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.sum(param**2)

        # Normalize losses per number of datapoints
        full_model_loss += loss.item() / y.shape[0]
        full_reg_loss += reg_loss / y.shape[0]

        # Unify regularization and model loss
        loss += alpha * reg_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

    else:
        # Losses were already adjusted, so now we just need to normalize per number of batches (and alpha)
        adjusted_model_loss = full_model_loss / len(dataloader)
        adjusted_reg_loss = alpha * full_reg_loss / len(dataloader)

        # Print and write info
        print(f"Train loss: {adjusted_model_loss}")
        print(f"\tReg Loss: {adjusted_reg_loss}")

        # Save data at specified path
        with open(path, "a+") as f:
            f.write(f"T:{adjusted_model_loss}\nR:{adjusted_reg_loss}\n")


def test_loop(
    path: str,
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable,
):
    """Evaluates a model on a given dataset. Supports Autoencoders, MLPs and RNNS.
    @param path: Path for the model performance (i.e. loss values) to be saved.
    @param dataloader: DataLoader containing the training data.
    @param model: The model to be trained.
    @param loss_fn: The loss function to be used. Must return a single number and only accept a prediction and target.
    @return: None.
    """
    # Set model to evaluation/inference mode
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            if isinstance(model, torch.nn.RNN):
                rnn_data = rnn_reshaping(X)
                states, _ = model(rnn_data)
                pred = create_rdms(torch.squeeze(rnn_unshaping(states, X.shape)))
            elif isinstance(model, (Autoencoder, RdmMlp)):
                pred = model(X)
            else:
                raise NotImplemented("This type of architecture is not yet supported.")
            test_loss += loss_fn(pred, y)

    # Write performance to file
    with open(path, "a+") as f:
        f.write(f"V:{test_loss / size}\n")

    # Adjust loss to be mean loss
    test_loss /= size
    print(f"Eval loss: {test_loss:>8f}")


def classic_training_loop(
    base_embedder: BaseEstimator, eeg_data: NDArray, kin_rdms: NDArray
) -> Tuple[float, NDArray]:
    """Uses a scikit learn method to compress/embedd data and train in with as much data as possible, then computes loss and training RDMs.

    @param base_embedder: A scikit-learn method implementing .fit() and .transform(), compressing from (X, X) to (X, 1).
    @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    @param kin_rdms: Pre-computed kinematics RDMs to compare against.
    @return: A training loss value and training RDMs of the employed method.
    """
    # Save time series locally, then compute rdms for participants
    accumulated_data = torch.empty(
        (
            eeg_data.shape[0] * eeg_data.shape[1],
            eeg_data.shape[2],
            eeg_data.shape[4],
        )
    )

    accumulated_data[:] = np.nan

    # Per phase, condition and time we want to fit the method
    for phase_idx in range(eeg_data.shape[1]):
        for condition_idx in range(eeg_data.shape[2]):
            for time_idx in range(eeg_data.shape[4]):
                base_embedder = base_embedder.fit(
                    eeg_data[:, phase_idx, condition_idx, :, time_idx]
                )

                # Afterwards transform per participant and create RDM
                for subject_idx in range(eeg_data.shape[0]):
                    accumulated_data[
                        subject_idx * eeg_data.shape[1] + phase_idx,
                        condition_idx,
                        time_idx,
                    ] = torch.as_tensor(
                        base_embedder.transform(
                            eeg_data[
                                subject_idx, phase_idx, condition_idx, :, time_idx
                            ].reshape(1, 16)
                        )
                    )

    assert torch.any(torch.isnan(accumulated_data)).item() is False

    accumulated_data = create_rdms(accumulated_data)

    loss = np.array([])
    for mat_eeg, mat_kin in zip(accumulated_data, kin_rdms):
        loss = np.append(loss, fro_loss(mat_eeg, mat_kin))

    return loss, accumulated_data
