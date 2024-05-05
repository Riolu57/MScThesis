import torch
from util.type_hints import DataShape


def create_eeg_data(eeg_data: torch.Tensor) -> torch.Tensor:
    """Unifies the Participant and Grasp phase dimensions, since the RDM MLP expects the data to be 4D.
    @param eeg_data: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    @return: EEG data of 4 dimensions: (Participants + Grasp phase x Condition x Channels x Time Points)
    """
    data_copy = eeg_data[:]
    return torch.reshape(
        data_copy,
        (
            eeg_data.shape[0] * eeg_data.shape[1],
            eeg_data.shape[2],
            eeg_data.shape[3],
            eeg_data.shape[4],
        ),
    )


def rnn_reshaping(eeg_data: torch.Tensor) -> torch.Tensor:
    """Reshapes 4D EEG data to be useful for an RNN. Reshaped, such that 'batches_first=True' condition is met.
    @param eeg_data: EEG data of 4 dimensions: (Participants + Grasp phase x Condition x Channels x Time Points)
    @return: EEG data of 3 dimensions: (Participants + Grasp phase + Condition x Channels x Time Points)
    """
    data_copy = eeg_data[:]
    return torch.reshape(
        data_copy,
        (
            data_copy.shape[0] * data_copy.shape[1],
            data_copy.shape[2],
            data_copy.shape[3],
        ),
    )


def rnn_unshaping(hidden_states: torch.Tensor, shape: DataShape) -> torch.Tensor:
    """Reshapes passed tensor to passed form. Used to change RNN hidden states output to original shape.

    @param hidden_states: Hidden states of RNN over batches and time.
    @param shape: Shape of old data.
    @return: Tensor of shape (Participants + Grasp phase x Condition x RNN hidden_neurons x Time Points)
    """
    data_copy = hidden_states[:]
    return torch.reshape(
        data_copy, (shape[0], shape[1], shape[2], hidden_states.shape[-1])
    )
