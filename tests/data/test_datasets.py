import unittest

import numpy as np
import torch

from util.data import AutoDataset, RDMDataset

from CONFIG import DTYPE_NP, DTYPE_TORCH


class AutoencoderDataset(unittest.TestCase):
    def setUp(self):
        # EEG data should be of shape (subject x grasping_phase x condition (objects x grasp_type) x channels x time)
        self.data = torch.rand(size=(30, 6, 100, 20, 200), dtype=DTYPE_TORCH)
        self.dataset = AutoDataset(self.data)

    def test_content(self):
        for idx, (inp, output) in enumerate(self.dataset):
            self.assertTrue(
                (inp == self.data[idx]).all(),
                "AutoDataset input is not equal to used data.",
            )
            self.assertTrue(
                (output == self.data[idx]).all(),
                "AutoDataset output is not equal to used data.",
            )
            self.assertTrue(
                (inp == output).all(),
                "AutoDataset input is not equal to AutoDataset output.",
            )

    def test_dtype(self):
        # Data type should be np.float64
        for idx, (inp, output) in enumerate(self.dataset):
            self.assertEqual(
                inp.dtype,
                DTYPE_TORCH,
                f"AutoDataset input is not of precision {DTYPE_TORCH}.",
            )
            self.assertEqual(
                output.dtype,
                DTYPE_TORCH,
                f"AutoDataset output is not of precision {DTYPE_TORCH}.",
            )

    def test_shape(self):
        for idx, (inp, output) in enumerate(self.dataset):
            self.assertTrue(
                (inp.shape == np.array(self.data.shape[1:])).all(),
                "AutoDataset input shape is incorrect",
            )
            self.assertTrue(
                (output.shape == np.array(self.data.shape[1:])).all(),
                "AutoDataset output shape is incorrect",
            )

    def test_type(self):
        for idx, (inp, output) in enumerate(self.dataset):
            self.assertTrue(
                isinstance(inp, type(torch.Tensor())),
                "AutoDataset input is not torch tensor.",
            )
            self.assertTrue(
                isinstance(output, type(torch.Tensor())),
                "AutoDataset input is not torch tensor.",
            )

    def tearDown(self) -> None:
        del self.data
        del self.dataset


class RepDisMatDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        # EEG data should be of shape (subject x grasping_phase x condition (objects x grasp_type) x channels x time)
        gen = np.random.default_rng(seed=None)
        self.data = torch.rand(size=(30, 6, 100, 20, 200), dtype=DTYPE_TORCH)
        self.dataset = RDMDataset(self.data, self.data)

        self.input = []
        self.output = []

        for inp, out in self.dataset:
            self.input.append(inp)
            self.output.append(out)

    def test_dtype(self):
        # Data type should be np.float64
        for idx, (inp, output) in enumerate(self.dataset):
            self.assertEqual(
                inp.dtype,
                DTYPE_TORCH,
                "RDMDataset input does not follow correct precision.",
            )
            self.assertEqual(
                output.dtype,
                DTYPE_TORCH,
                "RDMDataset output does not follow correct precision.",
            )

    def test_shape(self):
        for idx, (inp, output) in enumerate(self.dataset):
            self.assertTrue(
                (
                    inp.shape
                    == np.array(
                        [
                            self.data.shape[2],
                            self.data.shape[3],
                            self.data.shape[4],
                        ]
                    )
                ).all(),
                "RDMDataset input shape is incorrect",
            )
            self.assertTrue(
                (
                    output.shape == np.array([self.data.shape[2], self.data.shape[2]])
                ).all(),
                "RDMDataset output shape is incorrect",
            )

    def test_len(self):
        self.assertEqual(self.data.shape[0] * self.data.shape[1], len(self.dataset))

    def test_type(self):
        for idx, (inp, output) in enumerate(self.dataset):
            self.assertTrue(
                isinstance(inp, type(torch.Tensor())),
                "RDMDataset input is not torch tensor.",
            )
            self.assertTrue(
                isinstance(output, type(torch.Tensor())),
                "RDMDataset input is not torch tensor.",
            )

    def test_content_eeg(self):
        new_tensor_eeg = torch.empty(
            (
                self.data.shape[0] * self.data.shape[1],
                self.data.shape[2],
                self.data.shape[3],
                self.data.shape[4],
            )
        )
        new_tensor_eeg[:] = torch.nan

        for idx in range(self.data.shape[0]):
            new_tensor_eeg[
                idx * self.data.shape[1] : (idx + 1) * self.data.shape[1]
            ] = self.data[idx]

        self.assertFalse(
            torch.any(torch.isnan(new_tensor_eeg)),
            "Test reshaping went wrong.",
        )

        for idx, (inp, output) in enumerate(self.dataset):
            self.assertTrue(
                (inp == new_tensor_eeg[idx]).all(),
                "RDMDataset input is not equal to used data.",
            )

    def test_content_kin(self):
        new_tensor_kin = torch.empty(
            (
                self.data.shape[0] * self.data.shape[1],
                self.data.shape[2],
                self.data.shape[2],
            )
        )
        new_tensor_kin[:] = torch.nan

        ones = torch.ones(self.data.shape[2])

        # We want to create an RDM per participant and time-phase
        for idx_participant in range(self.data.shape[0]):
            for idx_phase in range(self.data.shape[1]):

                condition_timeserieses = torch.empty(0)

                # Go through the conditions to ensure equal treatment
                for idx_condition in range(self.data.shape[2]):
                    # Concatenate all channels and then calculate the correlation
                    concat_vec = self.data[
                        idx_participant, idx_phase, idx_condition
                    ].reshape(1, self.data.shape[3] * self.data.shape[4])

                    self.assertTrue(
                        (
                            concat_vec[0, : self.data.shape[4]]
                            == self.data[idx_participant, idx_phase, idx_condition, 0]
                        ).all(),
                        "Reshaping in creating test data went wrong.",
                    )

                    condition_timeserieses = torch.concatenate(
                        (condition_timeserieses, concat_vec)
                    )

                # Save RDM in correct position
                new_tensor_kin[
                    idx_participant * self.data.shape[1] + idx_phase
                ] = ones - torch.corrcoef(condition_timeserieses)

        self.assertFalse(
            torch.any(torch.isnan(new_tensor_kin)),
            "Test reshaping went wrong.",
        )

        for idx, (inp, output) in enumerate(self.dataset):
            self.assertTrue(
                (output == new_tensor_kin[idx]).all(),
                "RDMDataset output is not equal to used data.",
            )

    @classmethod
    def tearDownClass(self):
        del self.data
        del self.dataset


if __name__ == "__main__":
    unittest.main()
