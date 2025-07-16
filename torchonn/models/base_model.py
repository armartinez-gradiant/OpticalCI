"""
Base Model for PtONN-TESTS
"""

import torch
import torch.nn as nn
from typing import Optional, Union

class ONNBaseModel(nn.Module):
    """
    Base class for ONN models
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        super(ONNBaseModel, self).__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
    def reset_parameters(self):
        """Reset all parameters in the model."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()