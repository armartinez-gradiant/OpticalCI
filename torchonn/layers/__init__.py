"""
Layers module for PtONN-TESTS

Modern implementation of photonic layers compatible with current PyTorch versions.
"""

from .mzi_layer import MZILayer
from .mzi_block_linear import MZIBlockLinear

__all__ = [
    "MZILayer",
    "MZIBlockLinear",
]