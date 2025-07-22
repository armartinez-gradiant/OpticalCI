"""
Device configurations for PtONN-TESTS
"""

import torch
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

@dataclass
class DeviceConfig:
    """Configuration for photonic devices."""
    
    device: torch.device
    dtype: torch.dtype = torch.float32
    precision: str = "single"  # "single", "double", "half"
    noise_level: float = 0.0
    temperature: float = 300.0  # Kelvin
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.precision not in ["single", "double", "half"]:
            raise ValueError("precision must be 'single', 'double', or 'half'")
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be positive")
            
    @classmethod
    def from_device(cls, device: Union[str, torch.device], **kwargs) -> "DeviceConfig":
        """Create configuration from device."""
        if isinstance(device, str):
            device = torch.device(device)
        return cls(device=device, **kwargs)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "precision": self.precision,
            "noise_level": self.noise_level,
            "temperature": self.temperature,
        }

# Global default device configuration
_default_device_config: Optional[DeviceConfig] = None

def get_default_device() -> DeviceConfig:
    """Get default device configuration."""
    global _default_device_config
    if _default_device_config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _default_device_config = DeviceConfig(device=device)
    return _default_device_config

def set_default_device(config: Union[DeviceConfig, str, torch.device]) -> None:
    """Set default device configuration."""
    global _default_device_config
    if isinstance(config, (str, torch.device)):
        config = DeviceConfig.from_device(config)
    _default_device_config = config

def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "current_device": str(get_default_device().device),
    }
    
    if torch.cuda.is_available():
        info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info
