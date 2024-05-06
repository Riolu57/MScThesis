import unittest
import numpy as np
import torch

from CONFIG import DTYPE_NP, DTYPE_TORCH, FLOAT_TOLERANCE

from training.training_nn import RdmMlp, Autoencoder


class BatchProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        gen = np.random.default_rng(seed=None)
        # Only test whether a batch is processed correctly
        self.data = torch.as_tensor(
            gen.random((20, 16), dtype=DTYPE_NP), dtype=DTYPE_TORCH
        )
        self.rdm = RdmMlp(self.data.shape[1])
        self.auto = Autoencoder(self.data.shape[1])

    def test_batch_processing_rdm(self):
        batch_data = self.rdm.process(self.data)

        loop_data = torch.empty(0)
        for data_point in self.data:
            loop_data = torch.concatenate(
                (loop_data, self.rdm.process(data_point).reshape(1, 1))
            )

        self.assertTrue(((loop_data - batch_data) < FLOAT_TOLERANCE).all())

    def test_batch_processing_auto(self):
        batch_data = self.auto.process(self.data)

        loop_data = torch.empty(0)
        for data_point in self.data:
            loop_data = torch.concatenate(
                (
                    loop_data,
                    self.auto.process(data_point).reshape(1, self.data.shape[1]),
                )
            )

        self.assertTrue(((loop_data - batch_data) < FLOAT_TOLERANCE).all())
