import unittest

import numpy as np
import torch

from CONFIG import EEG_DATA_PATH, KIN_DATA_PATH, DTYPE_NP
from util.data import load_eeg_data, load_kinematics_data


class EEGLoading(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.data = load_eeg_data(EEG_DATA_PATH)

    def test_shape(self):
        # EEG data should be of shape (subject x grasping_phase x condition (objects x grasp_type) x channels x time)
        self.assertTrue(
            (self.data.shape == np.array([16, 6, 4 * 3, 16, 200])).all(),
            "EEG data shape is wrongly loaded.",
        )

    def test_dtype(self):
        # Data type should be np.float64
        self.assertEqual(self.data.dtype, DTYPE_NP, "EEG data is of wrong data type.")

    def test_type(self):
        self.assertTrue(
            isinstance(self.data, type(np.array([]))),
            "EEG data not loaded as np array.",
        )

    def test_nan(self):
        self.assertFalse(
            np.any(np.isnan(self.data)),
            "EEG data was loaded incomplete.",
        )

    @classmethod
    def tearDownClass(self):
        del self.data


class KinLoading(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.data = load_kinematics_data(KIN_DATA_PATH)

    def test_shape(self):
        # Kinematics data should be of shape
        # (subject*grasping_phase x condition (objects x grasp_type) x channels x time)
        self.assertTrue(
            (self.data.shape == np.array([16, 6, 4 * 3, 19, 200])).all(),
            "Kinematics data shape is wrongly loaded.",
        )

    def test_dtype(self):
        # Data type should be np.float64
        self.assertEqual(
            self.data.dtype, DTYPE_NP, "Kinematics data is of wrong shape."
        )

    def test_type(self):
        self.assertTrue(
            isinstance(self.data, type(np.array([]))),
            "Kinematics data not loaded as np array.",
        )

    def test_nan(self):
        self.assertFalse(
            np.any(np.isnan(self.data)),
            "Kinematics data was loaded incomplete.",
        )

    @classmethod
    def tearDownClass(self):
        del self.data


if __name__ == "__main__":
    unittest.main()
