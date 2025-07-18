"""
Device Configuration for TorchONN
=================================

Utilities for device configuration and management.
"""

import torch
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class DeviceConfig:
    """Configuration for compute devices."""
    device: torch.device
    precision: str = "float32"
    memory_fraction: float = 0.9
    
    @property
    def dtype(self) -> torch.dtype:
        """Get PyTorch dtype from precision string."""
        if self.precision == "float32":
            return torch.float32
        elif self.precision == "float64":
            return torch.float64
        elif self.precision == "float16":
            return torch.float16
        else:
            return torch.float32


def get_default_device() -> DeviceConfig:
    """Get default device configuration."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        precision = "float32"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        precision = "float32"
    else:
        device = torch.device("cpu")
        precision = "float32"
    
    return DeviceConfig(device=device, precision=precision)


def set_device_config(device: Union[str, torch.device], precision: str = "float32") -> DeviceConfig:
    """Set specific device configuration."""
    if isinstance(device, str):
        device = torch.device(device)
    
    return DeviceConfig(device=device, precision=precision)


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            info["cuda_devices"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
            })
    
    return info


# Default device configuration
DEFAULT_DEVICE_CONFIG = get_default_device()

__all__ = [
    'DeviceConfig',
    'get_default_device', 
    'set_device_config',
    'get_device_info',
    'DEFAULT_DEVICE_CONFIG'
]
