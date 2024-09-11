import torch
import unittest
from model.user.crossformer import CrossScaleParams, CrossformerParams
from model.user.crossformer import CrossformerEncoder
from .utils import timer, splitter


class TestEncoder(unittest.TestCase):
    @splitter
    @timer
    def test_build(self):
        cross_params = [
            CrossScaleParams(
                input_dim=1,
                output_dim=64,
                kernel_size=[4, 8, 16, 32],
                stride=4),
            CrossScaleParams(
                input_dim=64,
                output_dim=128,
                kernel_size=[2, 4],
                stride=2),
            CrossScaleParams(
                input_dim=128,
                output_dim=256,
                kernel_size=[2, 4],
                stride=2),
            CrossScaleParams(
                input_dim=256,
                output_dim=512,
                kernel_size=[2, 4],
                stride=2),
        ]
        transformer_params = [
            CrossformerParams(
                input_dim=64,
                group=7,
                n_layer=1
            ),
            CrossformerParams(
                input_dim=128,
                group=7,
                n_layer=1
            ),
            CrossformerParams(
                input_dim=256,
                group=7,
                n_layer=8
            ),
            CrossformerParams(
                input_dim=512,
                group=7,
                n_layer=6
            )
        ]

        encoder = CrossformerEncoder(cross_params, transformer_params)
        x = torch.randn(1, 1, 240, 240)
        features, shapes = encoder(x)
        print(list(map(lambda x: x.shape, features)))
        print(shapes)


if __name__ == '__main__':
    unittest.main()
