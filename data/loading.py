import os
from typing import Tuple

import h5py
import numpy as np
from numpy.typing import NDArray

from CONFIG import DTYPE_NP, EEG_CHANNELS


def load_kinematics_data(data_path: str) -> NDArray:
    """Loads a matlab matrix. Will split the data into different phases, Regions of Interest, grasps and subjects.
    @param data_path: The path to the Kinematics data.
    @return: Two matrices, representing the input and output for an RNN. The data matrix is of shape ()
    """
    """Loads kinematics data from Matlab matrix."""
    object_order = ["LC", "SC", "LS", "SS"]
    # Final data shape: 16 participants x 12 conditions x 19 channels x 1200 time steps
    all_data = np.empty((16, 4 * 3, 19, 1200), dtype=DTYPE_NP)

    # For each subject get folder and indices
    for subject_num in range(3, 19):
        # Concatenate along trials; to allow for proper indexing (even with missing data)
        subject_data = np.empty((1200, 0, 19))

        cur_folder = os.path.join(data_path, f"subject{subject_num}")
        indices = h5py.File(os.path.join(cur_folder, f"indexes_trials.mat"))

        # For each object load the data and cut the last two seconds
        for object_idx, object_type in enumerate(object_order):
            cur_file = h5py.File(
                os.path.join(
                    cur_folder, f"DataMeasures_{object_type}_S{subject_num}.mat"
                )
            )
            cur_data = np.array(
                cur_file.get("datasetLeapMotion"), order="F", dtype=DTYPE_NP
            )

            old_shape_subject_data = subject_data.shape

            # Remove last 2 seconds
            subject_data = np.concatenate((subject_data, cur_data[:1200, :, :]), axis=1)

            # Assert that data has been put in correct position
            assert (
                subject_data[:, old_shape_subject_data[1] :, :] == cur_data[:1200, :, :]
            ).all()

        # Check whether indices are disjoint and ordered correctly
        max_idx = -1
        # Load indices per object and grasp
        for object_idx, object_type in enumerate(object_order):
            cur_max = -1

            # Get the grasp types, average over them and save the data
            for grasp_idx, grasp_type in enumerate(
                ["power", "precision2", "precision5"]
            ):
                cur_indices = (
                    np.squeeze(
                        np.array(
                            indices[f"indexes_{object_type}_{grasp_type}"],
                            dtype="int32",
                        )
                    )
                    - 1  # MatLab indices start at 1, not 0
                )

                # Keep track of current min and max to update for next check
                if np.max(cur_indices) > cur_max:
                    cur_max = np.max(cur_indices)

                # We found misordered object order or overlapping trials
                if max_idx >= np.min(cur_indices):
                    raise ValueError("Indices are overlapping/Objects in wrong order")

                cur_grasp_data = subject_data[:, cur_indices, :]
                avg_data = np.mean(cur_grasp_data, axis=1)
                avg_data_transpose = avg_data.transpose()

                # Ensure that dimensions are swapped correctly
                assert (avg_data[4, :] == avg_data_transpose[:, 4]).all()

                all_data[
                    subject_num - 3, object_idx * 3 + grasp_idx, :, :
                ] = avg_data_transpose

            max_idx = cur_max

    final_data = np.empty((16, 6, 4 * 3, 19, 200), dtype=DTYPE_NP)
    final_data[:] = np.nan

    for idx in range(6):
        final_data[:, idx, :, :, :] = all_data[:, :, :, idx * 200 : (idx + 1) * 200]

    return final_data


def load_eeg_data(data_path: str) -> NDArray:
    """Loads a matlab matrix and expects the four arrays 'mean_MRCP_XX_precision5', with XX \in {LC, LS, SC, SS}.
    Will split the data into different phases, Regions of Interest, grasps and subjects.
    @param data_path: The path to the EEG data.
    @return: EEG data of 5 dimensions: (Participants x Grasp phase x Condition x Channels x Time Points)
    """
    # (1200ms x 64 electrodes x 16 subject x 3 grasp_types*4 objects)
    all_data = np.empty((1200, 64, 16, 12))

    data = h5py.File(data_path)

    # Needed order: ["LC", "SC", "LS", "SS"]
    for object_idx, object_type in enumerate(["LC", "SC", "LS", "SS"]):
        for grasp_idx, grasp_type in enumerate(["power", "precision2", "precision5"]):
            all_data[:, :, :, object_idx * 3 + grasp_idx] = np.array(
                data.get(f"mean_MRCP_{object_type}_{grasp_type}"),
                order="F",
                dtype=DTYPE_NP,
            )

    # SPLIT SIGNALS INTO EACH PART AS NECESSARY:
    #   - 4 Phases
    #       - The intervals are: [0:199]; [200:599]; [600:999]; [1000:1200]
    #   - 4 ROIs:              FC1, FCz, FC2;  C3, CZ, C4;    CP3, CP1, CPz, CP2, CP4;   P3, P1, Pz, P2, P4
    #       - Channels (0-63): [8, 45, 9];     [12, 13, 14];  [51, 18, 52, 19, 53];      [23, 56, 24, 57, 25]
    #   - 16 subjects
    #       - Each channel is its own subject
    #   - 4 Grasps:
    #       - Each *_DATA is its own grasp

    PHASE_it = [i for i in range(1200)]
    ROI_it = EEG_CHANNELS
    SUBJECT_it = [i for i in range(16)]
    GRASP_it = [i for i in range(4)]

    # Transpose to (subject, condition, channels, time)
    all_data = all_data.transpose(2, 3, 1, 0)
    roi_filtered_data = all_data[:, :, ROI_it, :]

    final_data = np.empty((16, 6, 4 * 3, 16, 200), dtype=DTYPE_NP)
    final_data[:] = np.nan

    # Create grasping phases (subject, grasping phases, condition, channels, time)
    for idx in range(6):
        final_data[:, idx, :, :, :] = roi_filtered_data[
            :, :, :, idx * 200 : (idx + 1) * 200
        ]

    return final_data


def split_data(data: np.array) -> Tuple[NDArray, NDArray, NDArray]:
    """Split data into training, testing and validation.
    Splits along the time dimension.

    @param data: Numpy array containing all inputs
    @return: Containg training, validation and testing data respectively.
    """

    test_size = int(0.1 * data.shape[0])

    train_data = np.empty(
        (data.shape[0] - 2 * test_size, *data.shape[1:]), dtype=DTYPE_NP
    )
    val_data = np.empty((test_size, *data.shape[1:]), dtype=DTYPE_NP)
    test_data = np.empty((test_size, *data.shape[1:]), dtype=DTYPE_NP)

    train_data[:] = np.nan
    val_data[:] = np.nan
    test_data[:] = np.nan

    train_data[:] = data[: train_data.shape[0]]
    val_data[:] = data[train_data.shape[0] : train_data.shape[0] + test_size]
    test_data[:] = data[
        train_data.shape[0] + test_size : train_data.shape[0] + 2 * test_size
    ]

    max_val = train_data.max()
    min_val = train_data.min()

    train_data = (train_data - min_val) / (max_val - min_val)
    val_data = (val_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    return train_data, val_data, test_data


def load_all_data(
    eeg_path: str, kin_path: str
) -> Tuple[Tuple[NDArray, NDArray, NDArray], Tuple[NDArray, NDArray, NDArray]]:
    """Loads and splits the EEG and kinematics data.
    @param eeg_path: Path to the EEG data.
    @param kin_path: Path to the Kinematics data.
    @return: Two tuples, containing the EEG and Kinematics training, validation and testing data respectively.
    """
    full_eeg_data = load_eeg_data(eeg_path)
    train_eeg_data, val_eeg_data, test_eeg_data = split_data(full_eeg_data)

    full_kinematics_data = load_kinematics_data(kin_path)
    train_kin, val_kin, test_kin = split_data(full_kinematics_data)

    return (train_eeg_data, val_eeg_data, test_eeg_data), (train_kin, val_kin, test_kin)
