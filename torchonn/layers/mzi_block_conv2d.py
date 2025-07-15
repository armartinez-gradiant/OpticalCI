import torch
import torch.nn as nn
from torchonn.op.ops import mzi_unitary_decompose

class MZIBlockConv2d(nn.Module):
    """Layer de bloque MZI para Conv2d."""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x