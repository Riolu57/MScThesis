import unittest

from networks.rdm_network import RdmMlp
from networks.autoencoder import Autoencoder


class AutoencoderArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.inp_dim = 15
        self.rdm = RdmMlp(self.inp_dim)
        self.auto = Autoencoder(self.inp_dim)

    def test_encoder_equivalence(self):
        for rdm_layer, auto_layer in zip(self.rdm.process, self.auto.process):
            self.assertEqual(
                repr(rdm_layer),
                repr(auto_layer),
                "Layers of architectures are not equivalent.",
            )

    def test_encoder_decoder_equivalence(self):
        for auto_layer_front, auto_layer_back in zip(
            self.auto.process[::2], self.auto.process[::2][::-1]
        ):
            self.assertEqual(
                type(auto_layer_front),
                type(auto_layer_back),
                "Layer type of encoder are not equivalent to decoder.",
            )

            self.assertEqual(
                auto_layer_front.in_features,
                auto_layer_back.out_features,
                "Features mismatched for encoder/decoder layers.",
            )

            self.assertEqual(
                auto_layer_front.out_features,
                auto_layer_back.in_features,
                "Features mismatched for encoder/decoder layers.",
            )

        for auto_activation_front, auto_activation_back in zip(
            self.auto.process[::-2][::-1], self.auto.process[::-2]
        ):
            self.assertEqual(
                repr(auto_activation_front),
                repr(auto_activation_back),
                "Activation layer between encoder/decoder not equivalent.",
            )
