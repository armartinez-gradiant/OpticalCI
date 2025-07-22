"""
Devices module for PtONN-TESTS

Device configurations and management for photonic computing.
"""

from .device_configs import DeviceConfig, get_default_device, set_default_device

__all__ = [
    "DeviceConfig",
    "get_default_device", 
    "set_default_device",
]
