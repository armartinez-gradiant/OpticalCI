import torch
from torchonn.op.ops import mzi_unitary_decompose

def test_identity_decompose():
    I = torch.eye(4)
    U = mzi_unitary_decompose(I)
    assert torch.allclose(U, I)