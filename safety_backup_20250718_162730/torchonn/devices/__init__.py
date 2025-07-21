"""
Dispositivos - TorchONN
======================

Módulo de configuración y gestión de dispositivos.
"""

try:
    from .device_configs import DeviceConfig, get_default_device
    __all__ = ['DeviceConfig', 'get_default_device']
except ImportError:
    # Fallback simple DeviceConfig
    import torch
    from dataclasses import dataclass
    
    @dataclass
    class DeviceConfig:
        device: torch.device
        precision: str = "float32"
    
    def get_default_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DeviceConfig(device=device)
    
    __all__ = ['DeviceConfig', 'get_default_device']
