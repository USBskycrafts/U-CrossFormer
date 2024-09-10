from model.user.cross_embedding import CrossScaleEmbedding
import unittest
import torch
import torch.nn as nn
from .utils import splitter, timer


class TestCrossScaleEmbedding(unittest.TestCase):
    @splitter
    def test_regular_shape(self):
        x = torch.Tensor(8, 16, 240, 240)

        model = CrossScaleEmbedding(16, 512)
        y = model(x)
        print(x.shape, y.shape)

        model = CrossScaleEmbedding(16, 512, reversed=True)
        x = model(y, y, x.shape)
        print(x.shape, y.shape)

        model = CrossScaleEmbedding(16, 32, [4, 8, 16, 32], 4)
        y = model(x)
        print(x.shape, y.shape)

        model = CrossScaleEmbedding(16, 32, [4, 8, 16, 32], 4, reversed=True)
        x = model(y, y, input_size=x.shape)
        print(x.shape, y.shape)

    @splitter
    def test_irregular_shape(self):
        x = torch.Tensor(8, 16, 37, 40)
        model = CrossScaleEmbedding(16, 512)
        y = model(x)
        print(x.shape, y.shape)

        model = CrossScaleEmbedding(16, 512, reversed=True)
        x = model(y, y, input_size=x.shape)
        print(x.shape, y.shape)

        model = CrossScaleEmbedding(32, 64, [4, 8, 16, 32], 4)
        x = torch.Tensor(8, 32, 91, 79)
        y = model(x)
        print(x.shape, y.shape)

        model = CrossScaleEmbedding(32, 64, [4, 8, 16, 32], 4, reversed=True)
        x = model(y, y, x.shape)
        print(x.shape, y.shape)


if __name__ == '__main__':
    unittest.main()
