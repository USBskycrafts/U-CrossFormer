import torch
import torch.nn as nn
from model.user.crossformer import CrossformerEncoder, CrossformerDecoder
from model.user.params import encoder_params, decoder_params
from typing import List
from tools.accuracy_tool import general_image_metrics
import unittest
from thop import profile_origin


class TestUserNet(unittest.TestCase):
    def test_forward(self):
        # Test forward pass of the user network
        self.encoder = CrossformerEncoder(*encoder_params.values())
        self.decoder = CrossformerDecoder(*decoder_params.values())
        self.loss = nn.L1Loss()
        x = torch.randn([8, 1, 240, 240])
        features, shapes = self.encoder(x)
        pred = self.decoder(features, shapes)
        loss = self.loss(pred, torch.randn([8, 1, 240, 240]))
        loss.backward()
        flops, params = profile_origin(self.encoder, inputs=(x,))
        print(f"{flops / 1e9} GFLOPs, {params / 1e6} MParams")
