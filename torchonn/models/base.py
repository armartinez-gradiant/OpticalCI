import torch
import torch.nn as nn

class ONNBaseModel(nn.Module):
    """Modelo base para ONN con esquema lineal."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)