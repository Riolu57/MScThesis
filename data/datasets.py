from typing import Tuple
from numpy.typing import NDArray

import numpy as np
import torch
from torch.utils.data import Dataset

from data.loading import split_data, load_all_data
from data.rdms import create_5D_rdms


class RDMDataset(Dataset):
    def __init__(self, eeg_data: NDArray, kinematics_data: NDArray):
        """Creates a dataset with EEG data of 4 dimensions (Participants x Conditions x Input channels x Time steps) as input, and kinematics RDMs as targets.

        @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @param kinematics_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        """
        self.labels = self.create_rdms(kinematics_data)
        self.data = self.copy_eep(eeg_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def create_rdms(kin_data: NDArray) -> torch.Tensor:
        """Creates Representation Dissimilarity Maps (RDMs) based off of the passed data.
        @param kin_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @return: 4D Tensor of RDMs: (Index x Grasp Phase x Condition x Condition)
        """
        accum = torch.empty(0)
        data_copy = torch.tensor(kin_data[:])
        for participant in data_copy:
            rdms = torch.empty(0)
            res = participant.reshape(
                participant.shape[0],
                participant.shape[1],
                participant.shape[2] * participant.shape[3],
            )
            for y in res:
                rdm = (torch.ones(participant.shape[1]) - torch.corrcoef(y)).reshape(
                    1, 1, participant.shape[1], participant.shape[1]
                )
                rdms = torch.concatenate((rdms, rdm), axis=1)
            accum = torch.concatenate((accum, rdms), 0)
        return accum

    @staticmethod
    def copy_eep(eeg_data: NDArray) -> torch.Tensor:
        """Copies NP array to Tensor.

        @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @return: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        """
        tensor_data = torch.as_tensor(eeg_data)
        return tensor_data


class AutoDataset(Dataset):
    def __init__(self, data):
        self.labels = data
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class RDMKinDataset(Dataset):
    def __init__(self, eeg_data: NDArray, kin_data: NDArray):
        """Creates a dataset with EEG data of 4 dimensions (Participants x Conditions x Input channels x Time steps) as input, and kinematics RDMs as targets.

        @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @param kinematics_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        """
        self.rdms = self.create_rdms(kin_data)
        self.data = self.copy_to_tensor(eeg_data)
        self.kin = self.copy_to_tensor(kin_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], (self.rdms[idx], self.kin[idx])

    @staticmethod
    def create_rdms(kin_data: NDArray) -> torch.Tensor:
        """Creates Representation Dissimilarity Maps (RDMs) based off of the passed data.
        @param kin_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
        @return: 4D Tensor of RDMs: (Index x Grasp Phase x Condition x Condition)
        """
        accum = torch.empty(0)
        data_copy = torch.tensor(kin_data[:])
        for participant in data_copy:
            rdms = torch.empty(0)
            res = participant.reshape(
                participant.shape[0],
                participant.shape[1],
                participant.shape[2] * participant.shape[3],
            )
            for y in res:
                rdm = (torch.ones(participant.shape[1]) - torch.corrcoef(y)).reshape(
                    1, 1, participant.shape[1], participant.shape[1]
                )
                rdms = torch.concatenate((rdms, rdm), axis=1)
            accum = torch.concatenate((accum, rdms), 0)
        return accum

    @staticmethod
    def copy_to_tensor(data: NDArray) -> torch.Tensor:
        """Copies NP array to Tensor.

        @param data: Data as NP array
        @return: Data as Torch tensor
        """
        tensor_data = torch.as_tensor(data)
        return tensor_data


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


def prepare_kin_eeg_data_rnn(
    eeg_data_path: str, kin_data_path: str
) -> Tuple[EEGKinDataset, EEGKinDataset, EEGKinDataset]:
    """Creates a dataset of training, validation and testing data to train for EEG -> Kinematics predictions. Suitable for RNNs.

    @param eeg_data_path: Path to the EEG data.
    @param kinematics_data_path: Path to the kinematics data.
    @return: Tuple containing training, validation and testing datasets.
    """
    return _prepare_rdm_datasets(eeg_data_path, kin_data_path, EEGKinDataset)


def prepare_eeg_emb_kin_data(
    eeg_data_path: str, kin_data_path: str
) -> Tuple[RDMKinDataset, RDMKinDataset, RDMKinDataset]:
    """Creates a dataset of training, validation and testing data to train for EEG -> Kinematics predictions and train
        the createed embeddings using RSA.

    @param eeg_data_path: Path to the EEG data.
    @param kinematics_data_path: Path to the kinematics data.
    @return: Tuple containing training, validation and testing datasets.
    """
    return _prepare_rdm_datasets(eeg_data_path, kin_data_path, RDMKinDataset)
