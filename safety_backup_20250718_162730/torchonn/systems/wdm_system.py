"""
WDM System - Sistema de Multiplexado por División de Longitud de Onda
====================================================================

Implementación de sistema WDM para comunicaciones fotónicas.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

class WDMSystem(nn.Module):
    """
    Sistema WDM (Wavelength Division Multiplexing).
    
    Permite multiplexar/demultiplexar múltiples canales ópticos
    en diferentes longitudes de onda.
    """
    
    def __init__(
        self,
        wavelengths: List[float],
        channel_spacing: float = 0.8e-9,  # 0.8 nm spacing
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.wavelengths = torch.tensor(wavelengths, device=device)
        self.n_channels = len(wavelengths)
        self.channel_spacing = channel_spacing
        self.device = device
        
        # Multiplexer/demultiplexer transfer matrices
        self.mux_matrix = nn.Parameter(
            torch.eye(self.n_channels, device=device) + 
            torch.randn(self.n_channels, self.n_channels, device=device) * 0.01
        )
        
        self.demux_matrix = nn.Parameter(
            torch.eye(self.n_channels, device=device) + 
            torch.randn(self.n_channels, self.n_channels, device=device) * 0.01
        )
        
        # Channel filters (simplified as learnable parameters)
        self.channel_filters = nn.Parameter(
            torch.ones(self.n_channels, device=device)
        )
        
    def multiplex(self, channel_signals: List[torch.Tensor]) -> torch.Tensor:
        """
        Multiplexar señales de múltiples canales.
        
        Args:
            channel_signals: Lista de tensores [batch_size, signal_length]
            
        Returns:
            multiplexed_signal: Tensor multiplexado [batch_size, signal_length]
        """
        if len(channel_signals) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(channel_signals)}")
        
        # Stack signals
        stacked_signals = torch.stack(channel_signals, dim=1)  # [batch, n_channels, signal_length]
        
        # Apply multiplexer matrix
        mux_output = torch.matmul(self.mux_matrix.unsqueeze(0), stacked_signals)
        
        # Sum across channels (wavelength multiplexing)
        multiplexed = mux_output.sum(dim=1)
        
        return multiplexed
    
    def demultiplex(self, multiplexed_signal: torch.Tensor) -> List[torch.Tensor]:
        """
        Demultiplexar señal WDM en canales individuales.
        
        Args:
            multiplexed_signal: Tensor [batch_size, signal_length]
            
        Returns:
            channel_signals: Lista de tensores demultiplexados
        """
        batch_size, signal_length = multiplexed_signal.shape
        
        # Expand signal for each channel
        expanded_signal = multiplexed_signal.unsqueeze(1).expand(-1, self.n_channels, -1)
        
        # Apply demultiplexer matrix
        demux_output = torch.matmul(self.demux_matrix.unsqueeze(0), expanded_signal)
        
        # Apply channel filters
        filtered_output = demux_output * self.channel_filters.view(1, -1, 1)
        
        # Convert to list of tensors
        channel_signals = [filtered_output[:, i, :] for i in range(self.n_channels)]
        
        return channel_signals
    
    def add_channel_crosstalk(self, signals: torch.Tensor, crosstalk_db: float = -30):
        """Añadir crosstalk entre canales"""
        crosstalk_linear = 10 ** (crosstalk_db / 20)
        crosstalk_matrix = torch.eye(self.n_channels, device=self.device) +                           torch.randn(self.n_channels, self.n_channels, device=self.device) * crosstalk_linear
        
        return torch.matmul(crosstalk_matrix.unsqueeze(0), signals)
    
    def get_channel_info(self) -> Dict:
        """Obtener información de los canales WDM"""
        return {
            'n_channels': self.n_channels,
            'wavelengths_nm': (self.wavelengths * 1e9).tolist(),
            'channel_spacing_nm': self.channel_spacing * 1e9,
            'frequency_range_thz': [
                (3e8 / (self.wavelengths.max() * 1e12)).item(),
                (3e8 / (self.wavelengths.min() * 1e12)).item()
            ]
        }

class WDMMultiplexer(WDMSystem):
    """Alias para compatibilidad hacia atrás"""
    pass

__all__ = ['WDMSystem', 'WDMMultiplexer']
