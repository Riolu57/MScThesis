import torch
from util.type_hints import *


def create_5D_rdms(data: DataConstruct) -> torch.Tensor:
    """Creates Representation Dissimilarity Maps (RDMs) based off of the passed data.
    @param data: Data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    @return: 3D Tensor of RDMs: (Index x Condition x Condition)
    """
    data_copy = torch.as_tensor(data[:])
    # Concatenate channels and combine participants and grasp phases
    data_copy = data_copy.reshape(
        data.shape[0] * data.shape[1], data.shape[2], data.shape[3] * data.shape[4]
    )
    return create_rdms(data_copy)


def create_rdms(data: torch.Tensor) -> torch.Tensor:
    """Creates correlation based RDMs based off of the given data.

    @param data: A numpy array containing the to-be-processed data, needs to be of shape (Z x X x Y)
    @return: An RDM corresponding to the given data. Will be of shape (Z x X x X)
    """

    accumulate = torch.empty(0)

    for elem in data:
        rdm = _create_rdm(elem)
        accumulate = torch.concatenate(
            (accumulate, torch.reshape(rdm, (1, rdm.shape[0], rdm.shape[0]))), 0
        )

    return accumulate


def _create_rdm(data: torch.Tensor) -> torch.Tensor:
    """Creates a correlation based RDM based off of the given data.

    @param data: A numpy array containing the to-be-processed data, needs to be of shape (X x Y)
    @return: An RDM corresponding to the given data. Will be of shape (X x X)
    """

    # Create RDM using the (12, X) matrix
    corr = torch.corrcoef(data)
    ones = torch.ones_like(corr)

    return ones - corr
