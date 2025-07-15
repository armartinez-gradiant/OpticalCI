import torch
from torchonn.layers import MZILayer, MZIBlockLinear

def test_mzi_layer():
    layer = MZILayer(phase=0.2)
    x = torch.ones(3,3)
    y = layer(x)
    assert y.shape == x.shape

def test_mzi_block_linear():
    layer = MZIBlockLinear(4,4)
    x = torch.randn(2,4)
    y = layer(x)
    assert y.shape == (2,4)