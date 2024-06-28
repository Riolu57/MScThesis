from util.type_hints import DataShape
from typing import Tuple

import torch


def rnn_reshaping(eeg_data: torch.Tensor) -> torch.Tensor:
    """Reshapes 6D EEG data to be useful for an RNN. Reshaped, such that 'batches_first=True' condition is met.
    @param eeg_data: EEG data of 4 dimensions: (Participants + Grasp phase x Condition x Channels x Time Points)
    @return: EEG data of 3 dimensions: (Participants + Grasp phase + Condition x Channels x Time Points)
    """
    data_copy = eeg_data[:]
    data_copy = data_copy.transpose(3, 4)
    return torch.reshape(
        data_copy,
        (
            data_copy.shape[0] * data_copy.shape[1] * data_copy.shape[2],
            data_copy.shape[3],
            data_copy.shape[4],
        ),
    )


def rnn_unshaping(hidden_states: torch.Tensor, shape: DataShape) -> torch.Tensor:
    """Reshapes passed tensor to passed form. Used to change RNN hidden states output to original shape.

    @param hidden_states: Hidden states of RNN over batches and time.
    @param shape: Shape of old data.
    @return: Tensor of shape (Participants x Grasp phase x Condition x RNN hidden_neurons x Time Points)
    """
    data_copy = hidden_states[:]
    data_copy = torch.reshape(
        data_copy, (shape[0], shape[1], shape[2], shape[4], hidden_states.shape[-1])
    )
    return data_copy.transpose(3, 4)


def cnn_unshaping(hidden_states: torch.Tensor, shape: DataShape) -> torch.Tensor:
    """Reshapes passed tensor to passed form. Used to change CNN hidden states output to near original shape.

    @param hidden_states: Hidden states of CNN over batches and time.
    @param shape: Shape of old data.
    @return: Tensor of shape (Participants x Grasp phase x Condition x CNN hidden_neurons x Reduced Time Points)
    """
    data_copy = hidden_states[:]
    data_copy = torch.reshape(
        data_copy,
        (shape[0], shape[1], shape[2], hidden_states.shape[1], hidden_states.shape[2]),
    )
    return data_copy


def adjust_5D_data(eeg_data: torch.Tensor) -> torch.Tensor:
    data_copy = eeg_data[:]
    return data_copy.reshape(
        data_copy.shape[0] * data_copy.shape[1],
        data_copy.shape[2],
        data_copy.shape[3],
        data_copy.shape[4],
    )


def reshape_to_3D(data: torch.Tensor) -> torch.Tensor:
    """Transforms data into (All, channel, time).

    @param data: 4D data of shape (Participant * Grasp Phase, Condition, Channel, Time)
    @return: Tensor with (Participant x Grasp Phase x Condition, Channel, Time)
    """
    return torch.reshape(
        data,
        (data.shape[0] * data.shape[1], data.shape[2], data.shape[3]),
    )


def unshape_from_3D(
    data: torch.Tensor, shape: Tuple[int, int, int, int, int]
) -> torch.Tensor:
    """Shapes 3D data to 5D.

    @param data: 3D data of shape (All, channel, time)
    @param shape: Tuple containing the goal shape.
    @return: 5D data of shape (Participant, Grasp Phase, Condition, Channel, Time)
    """
    return torch.reshape(data, shape)
