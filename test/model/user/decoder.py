import torch
import unittest
from model.user.crossformer import CrossScaleParams, CrossformerParams
from model.user.crossformer import CrossformerEncoder, CrossformerDecoder
from .utils import timer, splitter


class TestEncoder(unittest.TestCase):
    @splitter
    @timer
    def test_build(self):
        cross_params = [
            CrossScaleParams(
                input_dim=64,
                output_dim=1,
                kernel_size=[4, 8, 16, 32],
                stride=4),
            CrossScaleParams(
                input_dim=128,
                output_dim=64,
                kernel_size=[2, 4],
                stride=2),
            CrossScaleParams(
                input_dim=256,
                output_dim=128,
                kernel_size=[2, 4],
                stride=2),
            CrossScaleParams(
                input_dim=512,
                output_dim=256,
                kernel_size=[2, 4],
                stride=2),
        ]
        transformer_params = [
            CrossformerParams(
                input_dim=1,
                group=7,
                n_layer=1
            ),
            CrossformerParams(
                input_dim=64,
                group=7,
                n_layer=1
            ),
            CrossformerParams(
                input_dim=128,
                group=7,
                n_layer=8
            ),
            CrossformerParams(
                input_dim=256,
                group=7,
                n_layer=6
            )
        ]

        transformer_params.reverse()
        cross_params.reverse()

        decoder = CrossformerDecoder(cross_params, transformer_params)
        features = [torch.randn([1, 64, 60, 60]),
                    torch.randn([1, 128, 30, 30]),
                    torch.randn([1, 256, 15, 15]),
                    torch.randn([1, 512, 7, 7])]
        shapes = [torch.Size([1, 1, 240, 240]),
                  torch.Size([1, 64, 60, 60]),
                  torch.Size([1, 128, 30, 30]),
                  torch.Size([1, 256, 15, 15])]
        y = decoder(features, shapes)
        print(y.shape)


if __name__ == '__main__':
    unittest.main()
