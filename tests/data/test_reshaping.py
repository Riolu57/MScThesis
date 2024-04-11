import unittest
import numpy as np
import h5py
import torch

from CONFIG import EEG_DATA_PATH, DTYPE_NP, EEG_CHANNELS, DTYPE_TORCH

from util.data import (
    load_eeg_data,
    split_data,
    RDMDataset,
    prepare_rdm_data,
    create_rdms,
)
from training.training_classic import preprocess_pca_data
from training.training_nn import AUTOENCODER, RDM_MLP


class DataSplit(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        # EEG and Kinematics data should be of dtype DTYPE_NP and 5D array
        # EEG data should be of shape (subject x grasping_phase x condition (objects x grasp_type) x channels x time)
        gen = np.random.default_rng(seed=None)
        self.data = gen.random((30, 6, 100, 20, 200), dtype=DTYPE_NP)

        self.train, self.val, self.test = split_data(self.data)

    def test_type(self):
        self.assertTrue(
            isinstance(self.train, type(np.array([]))),
            "Training data not loaded as np array.",
        )

        self.assertTrue(
            isinstance(self.val, type(np.array([]))),
            "Validation data not loaded as np array.",
        )

        self.assertTrue(
            isinstance(self.test, type(np.array([]))),
            "Testing data not loaded as np array.",
        )

    def test_dtype(self):
        # Data type should be DTYPE_NP
        self.assertEqual(
            self.train.dtype, DTYPE_NP, "Training data is of wrong data type."
        )

        self.assertEqual(
            self.val.dtype, DTYPE_NP, "Validation data is of wrong data type."
        )

        self.assertEqual(
            self.test.dtype, DTYPE_NP, "Testing data is of wrong data type."
        )

    def test_shape(self):
        self.assertTrue(
            self.val.shape == self.test.shape,
            "Validation set is not the same shape as test set.",
        )

        self.assertEqual(
            self.val.shape[0] + self.test.shape[0] + self.train.shape[0],
            self.data.shape[0],
            "Data not split along first dimension/too much or too little data left after split.",
        )

        unsplit = np.concatenate((self.train, self.val, self.test), axis=0)
        self.assertTrue(
            unsplit.shape == self.data.shape,
            "Data not split along first dimension.",
        )

    def test_content(self):
        self.assertTrue(
            (self.train[:] == self.data[: self.train.shape[0]]).all(),
            "Train split not in data/got corrupted.",
        )
        self.assertTrue(
            (
                self.val[:]
                == self.data[
                    self.train.shape[0] : self.train.shape[0] + self.val.shape[0]
                ]
            ).all(),
            "Validation split not in data/got corrupted.",
        )
        self.assertTrue(
            (
                self.test[:] == self.data[self.train.shape[0] + self.val.shape[0] :]
            ).all(),
            "Test split not in data/got corrupted.",
        )

    @classmethod
    def tearDownClass(self):
        del self.data


class EEGReshaping(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.data = load_eeg_data(EEG_DATA_PATH)
        self.file = h5py.File(EEG_DATA_PATH)
        self.time_frames = 6

    def test_loading_reshaping(self):
        for object_idx, object_type in enumerate(["LC", "SC", "LS", "SS"]):
            for grasp_idx, grasp_type in enumerate(
                ["power", "precision2", "precision5"]
            ):
                cur_data = np.array(
                    self.file.get(f"mean_MRCP_{object_type}_{grasp_type}"),
                    order="F",
                    dtype=DTYPE_NP,
                )

                for grasp_phase in range(self.time_frames):
                    for participant in range(16):
                        for channel_idx in range(16):
                            self.assertTrue(
                                (
                                    self.data[
                                        participant,
                                        grasp_phase,
                                        object_idx * 3 + grasp_idx,
                                        channel_idx,
                                        :,
                                    ]
                                    == cur_data[
                                        grasp_phase * 200 : (grasp_phase + 1) * 200,
                                        EEG_CHANNELS,
                                        :,
                                    ][:, channel_idx, participant]
                                ).all(),
                                (
                                    f"Loading transformation goes wrong for EEG data with the following data:"
                                    f"\nObject: {object_type}, Grasp: {grasp_type}, Grasp phase: {grasp_phase},"
                                    f"Participant: {participant}, Channel: {channel_idx}."
                                ),
                            )

    @classmethod
    def tearDownClass(self):
        del self.data
        del self.file


class ClassicalReshaping(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        gen = np.random.default_rng(seed=None)
        # Generate data of shape (subject x grasping phase x condition x channels x time steps)
        self.data = gen.random((50, 10, 100, 20, 200), dtype=DTYPE_NP)
        self.pca_inp, self.pca_target = preprocess_pca_data(self.data, self.data)

    def test_pca_input_content(self):
        # We want PCA to be conducted over all participants and channels
        # Then we want to take the best eigenvector projection for all conditions and phases
        # Thus, the data needs to be reshaped to (grasping phase, condition x subject*channels x times steps)
        final_arr = np.empty(
            (
                self.data.shape[1],
                self.data.shape[2],
                self.data.shape[0] * self.data.shape[3],
                self.data.shape[4],
            )
        )
        final_arr[:] = np.nan

        for subject_idx in range(self.data.shape[0]):
            for phase_idx in range(self.data.shape[1]):
                for condition_idx in range(self.data.shape[2]):
                    for channel_idx in range(self.data.shape[3]):
                        final_arr[
                            phase_idx,
                            condition_idx,
                            subject_idx * self.data.shape[3] + channel_idx,
                            :,
                        ] = self.data[
                            subject_idx, phase_idx, condition_idx, channel_idx, :
                        ]

        self.assertFalse(np.any(np.isnan(final_arr)), "Final Arr creation went wrong")
        self.assertTrue(
            (self.pca_inp == final_arr).all(), "PCA preprocesses the data wrongly."
        )

    def test_pca_output_content(self):
        # Use RDMDataset, since we want a centralized kinematics data reshaping test
        RDMData = RDMDataset(self.data, self.data)

        for idx, (inp, out) in enumerate(RDMData):
            self.assertTrue(
                (self.pca_target == out).all(),
                "PCA Kin RDM creation is wrong, or RDMDataset Kin RDM computation fails. Check test to verify.",
            )

    def test_pca_input_shape(self):
        self.assertTrue(self.pca_inp.shape == None)

    def test_rdm_creation(self):
        pass


class AutoencoderReshaping(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        gen = np.random.default_rng(seed=None)
        # Generate data of shape (subject x grasping phase x condition x channels x time steps)
        self.data = torch.as_tensor(
            gen.random((2, 3, 4, 20, 5), dtype=DTYPE_NP), dtype=DTYPE_TORCH
        )
        self.model = AUTOENCODER(0)

    def test_shaping_equivalence(self):
        self.assertTrue(
            (
                self.model.unshape_data(
                    self.model.reshape_data(self.data), self.data.shape
                )
                == self.data
            ).all()
        )

    def test_shape(self):
        self.assertTrue(
            (
                self.model.reshape_data(self.data).shape
                == (
                    self.data.shape[0]
                    * self.data.shape[1]
                    * self.data.shape[2]
                    * self.data.shape[4],
                    self.data.shape[3],
                )
            )
        )

    def test_content_order(self):
        deformed_data = self.model.reshape_data(self.data)
        for idx_0 in range(self.data.shape[0]):
            for idx_1 in range(self.data.shape[1]):
                for idx_2 in range(self.data.shape[2]):
                    for idx_4 in range(self.data.shape[4]):
                        self.assertTrue(
                            (
                                self.data[idx_0, idx_1, idx_2, :, idx_4]
                                == deformed_data[
                                    idx_0
                                    * self.data.shape[1]
                                    * self.data.shape[2]
                                    * self.data.shape[4]
                                    + idx_1 * self.data.shape[2] * self.data.shape[4]
                                    + idx_2 * self.data.shape[4]
                                    + idx_4,
                                    :,
                                ]
                            ).all()
                        )


class RSAEmbeddingReshaping(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        gen = np.random.default_rng(seed=None)
        # Generate data of shape (subject * grasping phase x condition x channels x time steps),
        # since RDMDatasets are used
        self.data = torch.as_tensor(
            gen.random((2 * 3, 4, 20, 5), dtype=DTYPE_NP), dtype=DTYPE_TORCH
        )
        self.model = RDM_MLP(self.data.shape[2])

    def test_shaping_equivalence(self):
        self.assertTrue(
            (
                self.model.unshape_data(
                    self.model.reshape_data(self.data), self.data.shape
                )
                == self.data
            ).all()
        )

    def test_shape(self):
        self.assertTrue(
            (
                self.model.reshape_data(self.data).shape
                == (
                    self.data.shape[0] * self.data.shape[1] * self.data.shape[3],
                    self.data.shape[2],
                )
            )
        )

    def test_content_order(self):
        reshaped_data = self.model.reshape_data(self.data)
        for idx_0 in range(self.data.shape[0]):
            for idx_1 in range(self.data.shape[1]):
                for idx_3 in range(self.data.shape[3]):
                    self.assertTrue(
                        (
                            self.data[idx_0, idx_1, :, idx_3]
                            == reshaped_data[
                                idx_0 * self.data.shape[1] * self.data.shape[3]
                                + idx_1 * self.data.shape[3]
                                + idx_3,
                                :,
                            ]
                        ).all()
                    )
