from typing import Tuple
from numpy.typing import NDArray

import numpy as np
import torch
from torch.utils.data import Dataset

from data.loading import split_data, load_all_data
from data.rdms import create_kin_rdms
from data.reshaping import create_eeg_data


class RDMDataset(Dataset):
    def __init__(self, eeg_data: NDArray, kinematics_data: NDArray, rnn: bool = False):
        """Creates a dataset with EEG data of 4 dimensions (Participants x Conditions x Input channels x Time steps) as input, and kinematics RDMs as targets.

        @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @param kinematics_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @rnn: Whether the data will be used for an rnn.
        """
        self.labels = self.create_rdms(kinematics_data)
        self.data = self.process_eeg(eeg_data, rnn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def create_rdms(kin_data: NDArray) -> torch.Tensor:
        """Creates Representation Dissimilarity Maps (RDMs) based off of the passed data.
        @param kin_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @return: 3D Tensor of RDMs: (Index x Condition x Condition)
        """
        return create_kin_rdms(kin_data)

    @staticmethod
    def process_eeg(eeg_data: NDArray, rnn: bool = False) -> torch.Tensor:
        """Unifies the Participant and Grasp phase dimensions, since the RDM MLP expects the data to be 4D.

        @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @param rnn: Whether the eeg data will be used as input for an RNN.
        @return: EEG data of 4 dimensions: (Participants + Grasp phase x Condition x Channels x Time Points)
        """
        if rnn:
            return create_eeg_data(torch.as_tensor(eeg_data)).transpose(2, 3)
        else:
            return create_eeg_data(torch.as_tensor(eeg_data))


class AutoDataset(Dataset):
    def __init__(self, data):
        self.labels = data
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class EEGKinDataset(Dataset):
    def __init__(self, eeg_data: NDArray, kinematics_data: NDArray):
        """Creates a dataset with EEG data of 5 dimensions (Participants x Conditions x Input channels x Time steps) as input, and kinematics data as targets.

        @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @param kinematics_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        """
        self.data = torch.as_tensor(eeg_data)
        self.labels = torch.as_tensor(kinematics_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def _prepare_rdm_datasets(
    eeg_data_path: str, kinematics_data_path: str, dataset: Dataset, *args, **kwargs
) -> Tuple[Dataset, Dataset, Dataset]:
    """Creates a dataset of training, validation and testing data using the provided dataset class. Said class must accept 2 inputs, EEG and kinematics data.

    @param eeg_data_path: Path to the EEG data.
    @param kinematics_data_path: Path to the kinematics data.
    @param dataset: Torch Dataset class which accepts EEG and Kinematics data as arguments.
    @return: Tuple containing training, validation and testing datasets.
    """
    (train_eeg_data, val_eeg_data, test_eeg_data), (
        train_kin,
        val_kin,
        test_kin,
    ) = load_all_data(eeg_data_path, kinematics_data_path)

    return (
        dataset(train_eeg_data, train_kin, *args, **kwargs),
        dataset(val_eeg_data, val_kin, *args, **kwargs),
        dataset(test_eeg_data, test_kin, *args, **kwargs),
    )


def prepare_rdm_data(
    eeg_data_path: str, kinematics_data_path: str
) -> Tuple[RDMDataset, RDMDataset, RDMDataset]:
    """Creates a dataset of training, validation and testing data to train for RDM embeddings. Suitable for MLPs. Input data will be of shape (Participants + Grasp phase x Conditions x Input channel x Time steps).

    @param eeg_data_path: Path to the EEG data.
    @param kinematics_data_path: Path to the kinematics data.
    @return: Tuple containing training, validation and testing datasets.
    """

    return _prepare_rdm_datasets(eeg_data_path, kinematics_data_path, RDMDataset)


def prepare_rdm_data_rnn(
    eeg_data_path: str, kinematics_data_path: str
) -> Tuple[RDMDataset, RDMDataset, RDMDataset]:
    """Creates a dataset of training, validation and testing data to train for RDM embeddings. Suitable for RNNs. Input data will be of shape (Participants + Grasp phase + Conditions x Input channel x Time steps).

    @param eeg_data_path: Path to the EEG data.
    @param kinematics_data_path: Path to the kinematics data.
    @return: Tuple containing training, validation and testing datasets.
    """

    return _prepare_rdm_datasets(
        eeg_data_path, kinematics_data_path, RDMDataset, rnn=True
    )


def prepare_kin_eeg_data_rnn(
    eeg_data_path: str, kin_data_path: str
) -> Tuple[EEGKinDataset, EEGKinDataset, EEGKinDataset]:
    """Creates a dataset of training, validation and testing data to train for EEG -> Kinematics predictions. Suitable for RNNs.

    @param eeg_data_path: Path to the EEG data.
    @param kinematics_data_path: Path to the kinematics data.
    @return: Tuple containing training, validation and testing datasets.
    """
    return _prepare_rdm_datasets(eeg_data_path, kin_data_path, EEGKinDataset)