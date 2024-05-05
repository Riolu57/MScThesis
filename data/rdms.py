import torch
from numpy.typing import NDArray


def create_kin_rdms(kin_data: NDArray) -> torch.Tensor:
    """Creates Representation Dissimilarity Maps (RDMs) based off of the passed data.
    @param kin_data: Kinematics data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    @return: 3D Tensor of RDMs: (Index x Condition x Condition)
    """
    labels = torch.empty(0)

    for participant in kin_data:
        for grasp_phase in participant:
            # Concatenate all channel data to create data of shape (num_objects*num_grasps, num_channels*num_time_steps)
            rdm_data = grasp_phase.reshape(
                (grasp_phase.shape[0], grasp_phase.shape[1] * grasp_phase.shape[2])
            )

            cur_rdm = 1 - torch.corrcoef(torch.as_tensor(rdm_data))
            cur_rdm = cur_rdm.reshape(1, kin_data.shape[2], kin_data.shape[2])
            labels = torch.concatenate((labels, cur_rdm), 0)

    return labels


def create_rdms(data):
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


def _create_rdm(data):
    """Creates a correlation based RDM based off of the given data.

    @param data: A numpy array containing the to-be-processed data, needs to be of shape (X x Y)
    @return: An RDM corresponding to the given data. Will be of shape (X x X)
    """

    # Create RDM using the (12, X) matrix
    corr = torch.corrcoef(data)
    ones = torch.ones_like(corr)

    return ones - corr
