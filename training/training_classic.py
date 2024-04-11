from sklearn.decomposition import PCA
import torch
import numpy as np
from networks.loss import fro_loss
from util.data import create_kin_rdms, create_rdms


def preprocess_pca_data(eeg_data, kin_data):
    final_eeg_data = eeg_data[:]
    final_eeg_data = final_eeg_data.reshape(
        (
            eeg_data.shape[1],
            eeg_data.shape[2],
            eeg_data.shape[0] * eeg_data.shape[3],
            eeg_data.shape[4],
        )
    )

    final_kin_data = kin_data[:]
    # final_kin_data = final_kin_data.reshape()

    return final_eeg_data, None


def preprocess_testing_data(data):
    """Concatenates data along channels, allowing for decomposition over time per condition and participant.

    :param data: A numpy array containing all EEG data of shape (subject x condition x channels x time steps).
    :return: Data of shape (subject x conditions x time steps x channels).
    """
    # Concatenate channels and participants per time point to allow PCA over time
    result = data.transpose((0, 1, 3, 2))
    assert (data[4, 3, :, 10] == result[4, 3, :, 10]).all()
    return result


def preprocess_training_data(data):
    """Concatenates data along participants and channels, allowing for decomposition over time and per condition.

    :param data: A numpy array containing all EEG data of shape (subject x condition x channels x time steps).
    :return: Data of shape (conditions x time steps x data).
    """
    # Concatenate channels and participants per time point to allow PCA over time
    result = data.transpose((1, 3, 0, 2))
    result = result.reshape(
        (result.shape[0], result.shape[1], result.shape[2] * result.shape[3])
    )

    return result


def get_pca_data(eeg_data, kin_data):
    pass


def ana_pca(eeg_data_training, kin_data_training):
    # Generate PCA with 1 component; to allow for easy correlation down the lone
    pca = PCA(n_components=1, copy=True, whiten=False, svd_solver="full")

    # Save time serieses locally, then compute rdms for participants
    accumulated_data = torch.empty(
        (
            eeg_data_training.shape[0] * eeg_data_training.shape[1],
            eeg_data_training.shape[2],
            eeg_data_training.shape[4],
        )
    )

    accumulated_data[:] = np.nan

    # Per phase, condition and time we want to fit PCA
    for phase_idx in range(eeg_data_training.shape[1]):
        for condition_idx in range(eeg_data_training.shape[2]):
            for time_idx in range(eeg_data_training.shape[4]):
                pca = pca.fit(
                    eeg_data_training[:, phase_idx, condition_idx, :, time_idx]
                )

                # Afterwards transform per participant and create RDM
                for subject_idx in range(eeg_data_training.shape[0]):
                    accumulated_data[
                        subject_idx * eeg_data_training.shape[1] + phase_idx,
                        condition_idx,
                        time_idx,
                    ] = torch.as_tensor(
                        pca.transform(
                            eeg_data_training[
                                subject_idx, phase_idx, condition_idx, :, time_idx
                            ].reshape(1, 16)
                        )
                    )

    assert torch.any(torch.isnan(accumulated_data)).item() is False

    accumulated_data = create_rdms(accumulated_data)

    labels = create_kin_rdms(kin_data_training)

    loss = np.array([])
    for mat_eeg, mat_kin in zip(accumulated_data, labels):
        loss = np.append(loss, fro_loss(mat_eeg, mat_kin))

    return loss, accumulated_data, labels
