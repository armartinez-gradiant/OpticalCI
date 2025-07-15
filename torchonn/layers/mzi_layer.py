import torch
import torch.nn as nn

class MZILayer(nn.Module):
    """Interferómetro Mach-Zehnder parametrizado como capa de PyTorch."""
    def __init__(self, phase: float = 0.0):
        super().__init__()
        self.phase = nn.Parameter(torch.tensor(phase))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ejemplo: combinación lineal según fase
        return x * torch.cos(self.phase) + x * torch.sin(self.phase)