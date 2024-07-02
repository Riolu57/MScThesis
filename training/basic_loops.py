import numpy as np

from util.type_hints import *
from util.prettier import pcolors
from typing import Callable
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator

import torch

from networks.loss import fro_loss
from networks.util import get_rdms

from networks.rnn_emb_kin import RnnEmbKin
from networks.mlp_emb_kin import MlpEmbKin
from networks.cnn_emb_kin import CnnEmbKin
from networks.predictor import Predictor

from data.rdms import create_rdms


def nn_train_loop(
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
        norm_param = y.shape[0]

        # Reset gradients
        optimizer.zero_grad()

        # Compute prediction and loss based on architecture
        if isinstance(model, (MlpEmbKin, RnnEmbKin, CnnEmbKin)):
            mean, var, rdms = get_rdms(X, model)
            loss = loss_fn(mean, var, rdms, y)
        elif isinstance(model, (Predictor)):
            output = model(X)
            loss = torch.nn.MSELoss()(output, y)
        else:
            raise NotImplemented("This type of architecture is not yet supported.")

        # Compute regularization loss
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.sum(param**2)

        # Normalize losses per number of datapoints
        full_model_loss += loss.item() / norm_param
        full_reg_loss += reg_loss / norm_param

        # Unify regularization and model loss
        loss += alpha * reg_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

    else:
        # Losses were already adjusted, so now we just need to normalize per number of batches (and alpha)
        adjusted_model_loss = full_model_loss / len(dataloader)
        adjusted_reg_loss = full_reg_loss / len(dataloader)

        # Print and write info
        print(f"{pcolors.OKGREEN}Train loss: {adjusted_model_loss}{pcolors.ENDC}")
        print(f"\t{pcolors.OKCYAN}Reg Loss: {adjusted_reg_loss}{pcolors.ENDC}")

        # Save data at specified path
        with open(path, "a+") as f:
            f.write(f"T:{adjusted_model_loss}\nR:{adjusted_reg_loss}\n")


def nn_test_loop(
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

    full_model_loss = 0

    # Go through training data according to batches
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):
            norm_param = y.shape[0]

            # Compute prediction and loss based on architecture
            if isinstance(model, (MlpEmbKin, RnnEmbKin, CnnEmbKin)):
                mean, var, rdms = get_rdms(X, model)
                loss = loss_fn(mean, var, rdms, y)
            elif isinstance(model, (Predictor)):
                output = model(X)
                loss = torch.nn.MSELoss()(output, y)
            else:
                raise NotImplemented("This type of architecture is not yet supported.")

            # Normalize losses per number of datapoints
            full_model_loss += loss.item() / norm_param

    # Write performance to file
    with open(path, "a+") as f:
        f.write(f"V:{loss / len(dataloader)}\n")

    # Adjust loss to be mean loss
    loss /= len(dataloader)
    print(f"{pcolors.WARNING}Eval loss: {loss:>8f}{pcolors.ENDC}")

    return loss


def classic_training_loop(
    base_embedder: BaseEstimator,
    eeg_data: NDArray,
    kin_rdms: NDArray,
    fitting_data: [None | NDArray],
) -> Tuple[float, NDArray]:
    """Uses a scikit learn method to compress/embedd data and train in with as much data as possible, then computes loss and training RDMs.

    @param base_embedder: A scikit-learn method implementing .fit() and .transform(), compressing from (X, X) to (X, 1).
    @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    @param kin_rdms: Pre-computed kinematics RDMs to compare against.
    @param fitting_data: the data which will be embedded and returned. If None, eeg_data will be fit and returned.
    @return: A training loss value and training RDMs of the employed method.
    """
    # Save time series locally, then compute rdms for participants
    accumulated_data = torch.empty(
        (
            fitting_data.shape[0] * fitting_data.shape[1],
            fitting_data.shape[2],
            fitting_data.shape[4],
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
                for subject_idx in range(fitting_data.shape[0]):
                    accumulated_data[
                        subject_idx * fitting_data.shape[1] + phase_idx,
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

    accumulated_rdms = create_rdms(accumulated_data)

    loss = np.array([])
    for mat_eeg, mat_kin in zip(accumulated_rdms, kin_rdms):
        loss = np.append(loss, fro_loss(mat_eeg, mat_kin))

    return (
        loss,
        accumulated_data.reshape(
            fitting_data.shape[0],
            fitting_data.shape[1],
            fitting_data.shape[2],
            1,
            fitting_data.shape[4],
        ),
        accumulated_rdms,
    )
