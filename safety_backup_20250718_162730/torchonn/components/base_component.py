"""
BasePhotonicComponent - Clase Base para Componentes Fotónicos
============================================================

Clase base abstracta que define la interfaz común para todos los
componentes fotónicos en el framework TorchONN.

Autor: PtONN-TESTS Team
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class BasePhotonicComponent(nn.Module, ABC):
    """
    Clase base abstracta para todos los componentes fotónicos.
    
    Proporciona funcionalidad común y define la interfaz que deben
    implementar todos los componentes fotónicos del sistema.
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.wavelength = 1550e-9  # Longitud de onda por defecto (1550 nm)
        self.power_budget = {}  # Presupuesto de potencia del componente
        
    @abstractmethod
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante del campo óptico.
        
        Args:
            input_field: Campo óptico de entrada
            
        Returns:
            Campo óptico de salida
        """
        pass
    
    def set_wavelength(self, wavelength: float):
        """Establecer longitud de onda de operación"""
        self.wavelength = wavelength
        
    def get_parameters_count(self) -> int:
        """Obtener número total de parámetros entrenables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_power_consumption(self) -> float:
        """Calcular consumo de potencia estimado (a implementar por subclases)"""
        return 0.0
    
    def reset_parameters(self):
        """Reinicializar parámetros del componente"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class WaveguideComponent(BasePhotonicComponent):
    """Clase base para componentes basados en guías de onda"""
    
    def __init__(self, length: float = 1e-3, neff: float = 2.4, **kwargs):
        super().__init__(**kwargs)
        self.length = length  # Longitud en metros
        self.neff = neff      # Índice efectivo
        
    def propagation_phase(self) -> float:
        """Calcular fase de propagación"""
        k0 = 2 * np.pi / self.wavelength
        return k0 * self.neff * self.length


class ResonatorComponent(BasePhotonicComponent):
    """Clase base para componentes resonantes"""
    
    def __init__(self, radius: float = 5e-6, neff: float = 2.4, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius  # Radio en metros
        self.neff = neff      # Índice efectivo
        
    def free_spectral_range(self) -> float:
        """Calcular rango espectral libre"""
        ng = self.neff  # Aproximación: ng ≈ neff
        circumference = 2 * np.pi * self.radius
        return self.wavelength**2 / (ng * circumference)
    
    def resonance_wavelengths(self, m_start: int = 1, m_end: int = 10) -> np.ndarray:
        """Calcular longitudes de onda de resonancia"""
        circumference = 2 * np.pi * self.radius
        m_values = np.arange(m_start, m_end + 1)
        return circumference * self.neff / m_values
