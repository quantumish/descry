import torch
from descry import VisionTransformer, __version__


def test_version():
    assert __version__ == '0.1.0'

def test_transformer():
    vt = VisionTransformer()
    vt.forward(torch.random(10, 3, 256, 256))
