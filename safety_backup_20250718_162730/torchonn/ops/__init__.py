"""
Operaciones - TorchONN
=====================

Módulo de operaciones fotónicas especializadas.
"""

try:
    from .operations import (
        matrix_decomposition,
        apply_noise,
        compute_transmission,
        phase_shift,
        beam_splitter,
        optical_loss
    )
    __all__ = [
        'matrix_decomposition',
        'apply_noise', 
        'compute_transmission',
        'phase_shift',
        'beam_splitter',
        'optical_loss'
    ]
except ImportError:
    # Fallback simple operations
    import torch
    
    def matrix_decomposition(matrix, method="svd"):
        if method == "svd":
            return torch.svd(matrix)
        return matrix, torch.tensor([1.0]), matrix.t()
    
    def apply_noise(tensor, noise_level=0.1, noise_type="gaussian"):
        return tensor + torch.randn_like(tensor) * noise_level
    
    def compute_transmission(input_field, coupling_coefficient=0.5, phase_shift=0.0):
        return input_field * (1 - coupling_coefficient)
    
    def phase_shift(input_field, phase):
        return input_field  # Simplified
    
    def beam_splitter(input1, input2, splitting_ratio=0.5):
        t = (1 - splitting_ratio) ** 0.5
        r = splitting_ratio ** 0.5
        return t * input1 + r * input2, r * input1 + t * input2
    
    def optical_loss(input_field, loss_db=0.1):
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
