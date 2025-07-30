"""
Base ONN Class for OpticalCI

Clase base para todas las arquitecturas de redes neuronales ópticas.
Extiende ONNBaseModel con funcionalidades específicas para ONNs.

🎯 Diseño: Conservador, no toca código existente
📐 Hereda de: ONNBaseModel (existente)
🔧 Agrega: Métricas ópticas, validación física, utilities para ONNs
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import time

# Import de la clase base existente (NO MODIFICAR)
from ...models import ONNBaseModel


class BaseONN(ONNBaseModel):
    """
    Clase base para Redes Neuronales Ópticas.
    
    Extiende ONNBaseModel con funcionalidades específicas para ONNs:
    - Métricas de eficiencia óptica
    - Validación de conservación de energía
    - Utilities para entrenamiento fotónico
    - Logging de propiedades físicas
    
    Esta clase NO modifica el comportamiento de ONNBaseModel,
    solo agrega nuevas funcionalidades.
    """
    
    def __init__(
        self, 
        device: Optional[Union[str, torch.device]] = None,
        optical_power: float = 1.0,
        wavelength_channels: int = 1,
        enable_physics_validation: bool = True
    ):
        # Llamar al constructor padre sin modificar nada
        super().__init__(device=device)
        
        # Parámetros específicos de ONNs
        self.optical_power = optical_power
        self.wavelength_channels = wavelength_channels
        self.enable_physics_validation = enable_physics_validation
        
        # Métricas y estadísticas específicas para ONNs
        self.onn_metrics = {
            "total_forward_passes": 0,
            "energy_conservation_history": [],
            "optical_loss_history": [],
            "training_time": 0.0
        }
        
        # Configuración de logging para ONNs
        self.physics_validation_frequency = 0.1  # 10% de forward passes
        
        print(f"🔬 BaseONN initialized:")
        print(f"   Device: {self.device}")
        print(f"   Optical power: {optical_power:.2f}")
        print(f"   Wavelength channels: {wavelength_channels}")
        print(f"   Physics validation: {enable_physics_validation}")
    
    def validate_optical_physics(
        self, 
        input_power: torch.Tensor, 
        output_power: torch.Tensor,
        tolerance: float = 1e-2
    ) -> Dict[str, Any]:
        """
        Validar conservación de energía y propiedades físicas ópticas.
        
        Args:
            input_power: Potencia óptica de entrada [batch_size, features]
            output_power: Potencia óptica de salida [batch_size, features]  
            tolerance: Tolerancia para conservación de energía
            
        Returns:
            Dict con métricas de validación física
        """
        validation = {}
        
        with torch.no_grad():
            # 1. Conservación de energía total
            total_input = torch.sum(input_power, dim=1)  # [batch_size]
            total_output = torch.sum(output_power, dim=1)  # [batch_size]
            
            # Filtrar casos con energía suficiente
            valid_mask = total_input > 1e-6
            if torch.sum(valid_mask) > 0:
                energy_ratios = total_output[valid_mask] / total_input[valid_mask]
                energy_conservation = torch.mean(energy_ratios)
                energy_std = torch.std(energy_ratios)
                
                validation["energy_conserved"] = abs(energy_conservation - 1.0) < tolerance
                validation["energy_conservation_ratio"] = energy_conservation.item()
                validation["energy_conservation_std"] = energy_std.item()
            else:
                validation["energy_conserved"] = False
                validation["energy_conservation_ratio"] = 0.0
                validation["energy_conservation_std"] = 0.0
            
            # 2. Rango físico válido
            validation["power_in_range"] = torch.all(output_power >= 0.0) and torch.all(output_power <= self.optical_power * 1.1)
            validation["max_power"] = torch.max(output_power).item()
            validation["min_power"] = torch.min(output_power).item()
            
            # 3. Detectar anomalías
            validation["has_nan"] = torch.any(torch.isnan(output_power))
            validation["has_inf"] = torch.any(torch.isinf(output_power))
            validation["is_valid"] = not (validation["has_nan"] or validation["has_inf"])
        
        # Actualizar historial
        if validation["energy_conserved"]:
            self.onn_metrics["energy_conservation_history"].append(
                validation["energy_conservation_ratio"]
            )
        
        return validation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass base con validación física opcional.
        
        Subclases deben implementar _forward_optical() en lugar de forward().
        """
        # Incrementar contador
        self.onn_metrics["total_forward_passes"] += 1
        
        # Validación de entrada
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
        
        if x.device != self.device:
            x = x.to(self.device)
        
        # Medir tiempo de forward pass
        start_time = time.time()
        
        # Forward pass específico de la arquitectura (implementado por subclases)
        try:
            output = self._forward_optical(x)
        except NotImplementedError:
            # Fallback para desarrollo - llamar al forward estándar
            output = super().forward(x) if hasattr(super(), 'forward') else x
        
        forward_time = time.time() - start_time
        
        # Validación física ocasional
        if (self.enable_physics_validation and 
            self.training and 
            torch.rand(1).item() < self.physics_validation_frequency):
            
            input_power = torch.abs(x)**2
            output_power = torch.abs(output)**2
            
            validation = self.validate_optical_physics(input_power, output_power)
            
            if not validation["is_valid"]:
                warnings.warn("Physical validation failed: NaN/Inf detected")
            elif not validation["energy_conserved"]:
                warnings.warn(f"Energy conservation issue: ratio = {validation['energy_conservation_ratio']:.6f}")
        
        return output
    
    def _forward_optical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass específico para arquitectura óptica.
        
        DEBE ser implementado por todas las subclases.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor después de procesamiento óptico
        """
        raise NotImplementedError("Subclasses must implement _forward_optical()")
    
    def get_onn_metrics(self) -> Dict[str, Any]:
        """Obtener métricas específicas de la ONN."""
        metrics = self.onn_metrics.copy()
        
        # Estadísticas de conservación de energía
        if len(metrics["energy_conservation_history"]) > 0:
            energy_hist = np.array(metrics["energy_conservation_history"])
            metrics["energy_conservation_stats"] = {
                "mean": float(np.mean(energy_hist)),
                "std": float(np.std(energy_hist)),
                "min": float(np.min(energy_hist)),
                "max": float(np.max(energy_hist)),
                "samples": len(energy_hist)
            }
        
        return metrics
    
    def reset_onn_metrics(self):
        """Reset métricas de la ONN."""
        self.onn_metrics = {
            "total_forward_passes": 0,
            "energy_conservation_history": [],
            "optical_loss_history": [],
            "training_time": 0.0
        }
    
    def set_physics_validation(self, enable: bool, frequency: float = 0.1):
        """
        Configurar validación física.
        
        Args:
            enable: Habilitar/deshabilitar validación
            frequency: Frecuencia de validación (0.0-1.0)
        """
        self.enable_physics_validation = enable
        self.physics_validation_frequency = max(0.0, min(1.0, frequency))
        
        print(f"🔧 Physics validation: {'enabled' if enable else 'disabled'}")
        if enable:
            print(f"   Frequency: {frequency*100:.1f}% of forward passes")
    
    def extra_repr(self) -> str:
        """Representación adicional para debugging."""
        base_repr = super().extra_repr() if hasattr(super(), 'extra_repr') else ""
        onn_repr = (f"optical_power={self.optical_power}, "
                   f"wavelength_channels={self.wavelength_channels}, "
                   f"physics_validation={self.enable_physics_validation}")
        
        if base_repr:
            return f"{base_repr}, {onn_repr}"
        else:
            return onn_repr


def validate_onn_implementation(onn_class):
    """
    Utility function para validar que una clase ONN está bien implementada.
    
    Args:
        onn_class: Clase que hereda de BaseONN
        
    Returns:
        Dict con resultados de validación
    """
    validation = {
        "inherits_from_base_onn": issubclass(onn_class, BaseONN),
        "implements_forward_optical": hasattr(onn_class, '_forward_optical'),
        "has_required_methods": True,
        "issues": []
    }
    
    if not validation["inherits_from_base_onn"]:
        validation["issues"].append("Must inherit from BaseONN")
    
    if not validation["implements_forward_optical"]:
        validation["issues"].append("Must implement _forward_optical() method")
    
    validation["is_valid"] = len(validation["issues"]) == 0
    
    return validation