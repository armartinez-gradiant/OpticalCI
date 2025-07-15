import torch
import torch.nn as nn
from torchonn.op.ops import mzi_unitary_decompose

class MZIBlockLinear(nn.Module):
    """Layer de bloque MZI para Linear."""
    def __init__(self, in_features, out_features, miniblock=4, mode="usv", device=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.miniblock = miniblock
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight
        u = mzi_unitary_decompose(w)
        return torch.matmul(x, u.t())