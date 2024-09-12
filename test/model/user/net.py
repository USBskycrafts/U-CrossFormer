import torch
import torch.nn as nn
from model.user.crossformer import CrossformerEncoder, CrossformerDecoder
from .params import encoder_params, decoder_params
from typing import List
from tools.accuracy_tool import general_image_metrics
import unittest
from thop import profile
from tools.init_tool import result
from tensorboardX import SummaryWriter


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.encoder = CrossformerEncoder(*encoder_params.values())
        self.decoder = CrossformerDecoder(*decoder_params.values())
        self.loss = nn.L1Loss()

    def forward(self, x):
        features, shapes = self.encoder(x)
        pred = self.decoder(features, shapes)
        loss = self.loss(pred, torch.randn([8, 1, 240, 240]))
        return loss


class TestUserNet(unittest.TestCase):
    def test_forward(self):
        # Test forward pass of the user network
        x = torch.randn([8, 2, 240, 240])
        model = Net()
        loss = model(x)
        loss.backward()
        flops, params = profile(model, inputs=(x,))
        writer = SummaryWriter("test/model/user/tensorboard")

        print(f"{flops / 1e9} GFLOPs, {params / 1e6} MParams")
        writer.add_graph(model, x)
