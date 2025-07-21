"""
Photonic Operations for TorchONN
===============================

Core operations for photonic neural networks.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def matrix_decomposition(
    matrix: torch.Tensor, 
    method: str = "svd"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose a matrix using various methods.
    
    Args:
        matrix: Input matrix to decompose
        method: Decomposition method ("svd", "qr", "lu")
    
    Returns:
        Decomposed matrices (U, S, V) for SVD or equivalent
    """
    if method == "svd":
        U, S, V = torch.svd(matrix)
        return U, S, V
    elif method == "qr":
        Q, R = torch.qr(matrix)
        # For compatibility, return Q, diag(R), identity
        return Q, torch.diag(R), torch.eye(matrix.size(1), device=matrix.device)
    else:
        raise ValueError(f"Unknown decomposition method: {method}")


def apply_noise(
    tensor: torch.Tensor,
    noise_level: float = 0.1,
    noise_type: str = "gaussian"
) -> torch.Tensor:
    """
    Apply noise to a tensor.
    
    Args:
        tensor: Input tensor
        noise_level: Noise strength (0-1)
        noise_type: Type of noise ("gaussian", "uniform")
    
    Returns:
        Noisy tensor
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(tensor) * noise_level
    elif noise_type == "uniform":
        noise = (torch.rand_like(tensor) - 0.5) * 2 * noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return tensor + noise


def compute_transmission(
    input_field: torch.Tensor,
    coupling_coefficient: float = 0.5,
    phase_shift: float = 0.0
) -> torch.Tensor:
    """
    Compute transmission through a photonic element.
    
    Args:
        input_field: Input optical field
        coupling_coefficient: Coupling strength (0-1)
        phase_shift: Phase shift in radians
    
    Returns:
        Transmitted field
    """
    transmission = torch.sqrt(1 - coupling_coefficient**2)
    phase_factor = torch.exp(1j * torch.tensor(phase_shift))
    
    # For real tensors, just apply transmission
    if not torch.is_complex(input_field):
        return input_field * transmission
    else:
        return input_field * transmission * phase_factor


def phase_shift(
    input_field: torch.Tensor,
    phase: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Apply phase shift to optical field.
    
    Args:
        input_field: Input field
        phase: Phase shift in radians
    
    Returns:
        Phase-shifted field
    """
    if isinstance(phase, (int, float)):
        phase = torch.tensor(phase, device=input_field.device)
    
    # For real inputs, convert to complex
    if not torch.is_complex(input_field):
        complex_field = input_field.to(torch.complex64)
    else:
        complex_field = input_field
    
    phase_factor = torch.exp(1j * phase)
    return complex_field * phase_factor


def beam_splitter(
    input1: torch.Tensor,
    input2: torch.Tensor,
    splitting_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate a beam splitter.
    
    Args:
        input1: First input beam
        input2: Second input beam  
        splitting_ratio: Splitting ratio (0-1)
    
    Returns:
        Two output beams
    """
    t = torch.sqrt(1 - splitting_ratio)  # Transmission
    r = torch.sqrt(splitting_ratio)      # Reflection
    
    output1 = t * input1 + r * input2
    output2 = r * input1 + t * input2
    
    return output1, output2


def optical_loss(
    input_field: torch.Tensor,
    loss_db: float = 0.1
) -> torch.Tensor:
    """
    Apply optical loss.
    
    Args:
        input_field: Input field
        loss_db: Loss in dB
    
    Returns:
        Attenuated field
    """
    loss_linear = 10 ** (-loss_db / 20)
    return input_field * loss_linear


__all__ = [
    'matrix_decomposition',
    'apply_noise',
    'compute_transmission', 
    'phase_shift',
    'beam_splitter',
    'optical_loss'
]
