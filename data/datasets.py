from CONFIG import DTYPE_TORCH

from typing import Tuple
from util.type_hints import DataConstruct

import torch
from torch.utils.data import Dataset

from data.rdms import create_5D_rdms
from data.reshaping import reshape_to_3D, adjust_5D_data


def copy_as_tensor(data: DataConstruct) -> torch.Tensor:
    """Casts data as torch Tensor.

    @param data: Some array-like construct
    @return: Data as tensor.
    """
    return torch.as_tensor(data, dtype=DTYPE_TORCH)


class EEGRDMDataset(Dataset):
    def __init__(self, eeg_data, kin_data):
        self.eeg = adjust_5D_data(copy_as_tensor(eeg_data))
        self.rdms = create_5D_rdms(copy_as_tensor(kin_data))

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.rdms[idx]


def prepare_eeg_rdm_data(
    eeg_data: Tuple[DataConstruct, DataConstruct, DataConstruct],
    kin_data: Tuple[DataConstruct, DataConstruct, DataConstruct],
) -> tuple[EEGRDMDataset, EEGRDMDataset, EEGRDMDataset]:
    """Creates a dataset which uses EEG data as input and associated kinematics RDMs as outputs.

    @param eeg_data: Loaded EEG data.
    @param kin_data: Loaded Kinematics data.
    @return: Train, Validation and Test data.
    """
    train_eeg, val_eeg, test_eeg = eeg_data
    train_kin, val_kin, test_kin = kin_data

    return (
        EEGRDMDataset(train_eeg, train_kin),
        EEGRDMDataset(val_eeg, val_kin),
        EEGRDMDataset(test_eeg, test_kin),
    )


class EmbKinDataset(Dataset):
    def __init__(self, emb_data: DataConstruct, kin_data: DataConstruct):
        self.emb = copy_as_tensor(emb_data)
        self.kin = copy_as_tensor(kin_data)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return self.emb[idx], self.kin[idx]


def _prepare_emb_kin_data(
    emb_data: Tuple[DataConstruct, DataConstruct, DataConstruct],
    kin_data: Tuple[DataConstruct, DataConstruct, DataConstruct],
) -> tuple[EmbKinDataset, EmbKinDataset, EmbKinDataset]:
    """Creates a dataset which uses embedded EEG data as input and associated kinematics data as outputs.

    @param emb_data: Embedded EEG data.
    @param kin_data: Loaded Kinematics data.
    @return: Train, Validation and Test data.
    """
    train_emb, val_emb, test_emb = emb_data
    train_kin, val_kin, test_kin = kin_data

    return (
        EmbKinDataset(train_emb, train_kin),
        EmbKinDataset(val_emb, val_kin),
        EmbKinDataset(test_emb, test_kin),
    )


def prepare_model_emb_kin_data(
    eeg_data: Tuple[DataConstruct, DataConstruct, DataConstruct],
    kin_data: Tuple[DataConstruct, DataConstruct, DataConstruct],
    model: torch.nn.Module,
) -> tuple[EmbKinDataset, EmbKinDataset, EmbKinDataset]:
    """Creates data from EEG emb. to Kin.

    @param eeg_data: Loaded EEG data.
    @param kin_data: Loaded Kin data.
    @param model: Torch network reducing the dimensionality. Needs to return 5D data.
    @return: Train, Val and Test data.
    """
    full_eeg_data = []
    full_kin_data = []
    model.eval()
    for data in eeg_data:
        full_eeg_data.append(model(adjust_5D_data(copy_as_tensor(data)))[2].detach())
    for data in kin_data:
        full_kin_data.append(adjust_5D_data(copy_as_tensor(data)))
    return _prepare_emb_kin_data(full_eeg_data, full_kin_data)
