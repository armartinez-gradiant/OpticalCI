"""
Operations module for PtONN-TESTS

Basic operations for photonic computing.
""" 

from .operations import (
    matrix_decomposition,
    phase_to_unitary,
    unitary_to_phase,
    apply_noise,
    thermal_phase_shift,
)

__all__ = [
    "matrix_decomposition",
    "phase_to_unitary", 
    "unitary_to_phase",
    "apply_noise",
    "thermal_phase_shift",
]