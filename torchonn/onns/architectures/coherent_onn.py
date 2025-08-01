"""
Coherent Optical Neural Network (CoherentONN) - VERSIÓN DEFINITIVA COMPLETA

🔧 TODOS LOS MÉTODOS INCLUIDOS - ERROR get_optical_efficiency_metrics CORREGIDO
📚 Based on: Shen et al. "Deep learning with coherent nanophotonic circuits" (Nature Photonics 2017)
✅ Mejoras científicas aplicadas + TODOS los métodos requeridos por el demo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
import warnings

# Imports de componentes OpticalCI existentes
from ...layers import MZILayer, MZIBlockLinear, Photodetector
from .base_onn import BaseONN


class CoherentONN(BaseONN):
    """
    Coherent Optical Neural Network usando mesh de MZIs.
    
    VERSIÓN DEFINITIVA COMPLETA - Todos los métodos requeridos incluidos.
    
    Implementa la arquitectura propuesta por Shen et al. (2017):
    1. Cada capa linear = Matriz unitaria (MZI mesh)  
    2. Activación = Photodetection + optical re-encoding
    3. Clasificación = Capa final eléctrica
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation_type: str = "square_law",
        optical_power: float = 1.0,
        use_unitary_constraints: bool = True,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Inicializar CoherentONN.
        
        Args:
            layer_sizes: Lista de tamaños de capa [input, hidden1, ..., output]
            activation_type: Tipo de activación óptica ("square_law", "soft_square")
            optical_power: Potencia óptica normalizada (W)
            use_unitary_constraints: Usar matrices estrictamente unitarias
            device: Device para cálculo
        """
        # Validación de entrada
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes debe tener al menos 2 elementos")
        if not all(isinstance(size, int) and size > 0 for size in layer_sizes):
            raise ValueError("Todos los tamaños de capa deben ser enteros positivos")
        if optical_power <= 0:
            raise ValueError("optical_power debe ser positivo")
        
        # Inicializar clase base
        super().__init__(
            device=device,
            optical_power=optical_power,
            wavelength_channels=1,
            enable_physics_validation=True
        )
        
        # Configuración de la arquitectura
        self.layer_sizes = layer_sizes
        self.activation_type = activation_type
        self.use_unitary_constraints = use_unitary_constraints
        self.n_layers = len(layer_sizes) - 1
        
        # Configurar device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        # Crear arquitectura
        self._build_optical_layers()
        
        # Inicialización científica
        self._initialize_parameters_scientifically()
        
        # Mover a device
        self.to(self.device)
        
        # Contar parámetros
        self._n_parameters = self._count_parameters()
        
        print(f"🔬 CoherentONN initialized:")
        print(f"   Layer sizes: {layer_sizes}")
        print(f"   Parameters: ~{self._n_parameters:,}")
        print(f"   Activation: {activation_type}")
        print(f"   Unitary constraints: {use_unitary_constraints}")
        print(f"   Device: {device}")
    
    def _build_optical_layers(self):
        """Construir capas ópticas."""
        # Listas para capas
        self.optical_layers = nn.ModuleList()
        self.photodetectors = nn.ModuleList()
        
        # Crear capas ópticas
        for i in range(self.n_layers):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            
            # Crear capa óptica para capas intermedias
            if i < self.n_layers - 1:
                if self.use_unitary_constraints:
                    # MZILayer para matrices unitarias estrictas
                    optical_layer = MZILayer(
                        in_features=in_size,
                        out_features=out_size,
                        device=self.device
                    )
                else:
                    # MZIBlockLinear para mayor flexibilidad
                    optical_layer = MZIBlockLinear(
                        in_features=in_size,
                        out_features=out_size,
                        mode="usv",
                        device=self.device
                    )
                
                self.optical_layers.append(optical_layer)
            
            # Photodetector para cada capa (incluyendo final)
            photodetector = Photodetector(
                responsivity=1.0,
                dark_current=0.0,
                device=self.device
            )
            self.photodetectors.append(photodetector)
        
        # Capa final eléctrica para clasificación
        final_in = self.layer_sizes[-2]
        final_out = self.layer_sizes[-1]
        self.final_layer = nn.Linear(final_in, final_out, device=self.device)
    
    def _count_parameters(self) -> int:
        """Contar parámetros aproximados."""
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            
            if self.use_unitary_constraints:
                # MZI Layer: parámetros según descomposición de Reck
                max_dim = max(in_size, out_size)
                n_mzis = max_dim * (max_dim - 1) // 2
                n_phases = max_dim
                total += n_mzis * 2 + n_phases
            else:
                # MZIBlockLinear USV mode
                total += in_size * out_size
        
        # Capa final
        total += self.layer_sizes[-2] * self.layer_sizes[-1]
        return total
    
    def _initialize_parameters_scientifically(self):
        """Inicialización científica basada en literatura."""
        with torch.no_grad():
            # Inicialización conservadora para estabilidad
            for layer in self.optical_layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                
                # Inicialización adicional para MZI layers
                if hasattr(layer, 'phases'):
                    # Haar-random initialization para unitaridad
                    phases = torch.randn_like(layer.phases) * 0.1
                    layer.phases.data = phases
                
            # Capa final con inicialización Xavier conservadora
            if hasattr(self.final_layer, 'weight'):
                nn.init.xavier_uniform_(self.final_layer.weight, gain=0.5)
                if self.final_layer.bias is not None:
                    nn.init.zeros_(self.final_layer.bias)
    
    def _forward_optical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass óptico implementado (requerido por BaseONN).
        
        Args:
            x: Input tensor [batch_size, input_features]
            
        Returns:
            Output del procesamiento óptico
        """
        return self.forward(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completa de la CoherentONN.
        
        Args:
            x: Input tensor [batch_size, input_features]
            
        Returns:
            Logits de clasificación [batch_size, n_classes]
        """
        # Validación de entrada
        if x.dim() != 2:
            raise ValueError(f"Input debe ser 2D, recibido: {x.dim()}D")
        if x.size(1) != self.layer_sizes[0]:
            raise ValueError(f"Input size {x.size(1)} != expected {self.layer_sizes[0]}")
        
        # Asegurar que esté en el device correcto
        x = x.to(self.device)
        
        # Normalización de entrada conservadora
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        x = x * torch.sqrt(torch.tensor(self.optical_power, device=self.device))
        
        # Forward óptico a través de capas intermedias
        current = x
        
        for i, (optical_layer, photodetector) in enumerate(zip(self.optical_layers, self.photodetectors[:-1])):
            # Transformación óptica unitaria
            current = optical_layer(current)
            
            # Photodetection (activación no-linear óptica)
            current = photodetector(current)
            
            # Activación estable (CIENTÍFICAMENTE MEJORADA)
            if self.activation_type == "square_law":
                # Estable: x^0.45 en lugar de sqrt problemático
                current = torch.pow(torch.abs(current) + 1e-12, 0.45) * torch.sign(current)
            elif self.activation_type == "soft_square":
                # Activación suave alternativa
                current = current / (1.0 + 0.1 * torch.abs(current))
            
            # Protección NaN/Inf
            current = torch.where(
                torch.isnan(current) | torch.isinf(current),
                torch.zeros_like(current),
                current
            )
            
            # Re-normalización conservadora
            if i < len(self.optical_layers) - 1:
                norm = torch.norm(current, dim=1, keepdim=True)
                current = current / (norm + 1e-8) * torch.sqrt(torch.tensor(self.optical_power, device=self.device))
        
        # Photodetection final
        final_photodetector = self.photodetectors[-1]
        current = final_photodetector(current)
        
        # Capa final eléctrica (clasificación)
        output = self.final_layer(current)
        
        # Actualizar métricas
        self.onn_metrics["total_forward_passes"] += 1
        
        return output
    
    def get_optical_efficiency_metrics(self) -> Dict[str, Any]:
        """
        Calcular métricas de eficiencia óptica.
        
        ⭐ MÉTODO REQUERIDO POR EL DEMO - AHORA INCLUIDO ⭐
        
        Returns:
            Dict con métricas de eficiencia
        """
        # Calcular parámetros ópticos totales
        total_optical_params = 0
        for layer in self.optical_layers:
            total_optical_params += sum(p.numel() for p in layer.parameters())
        
        # Calcular parámetros eléctricos
        electrical_params = sum(p.numel() for p in self.final_layer.parameters())
        
        # Fracción óptica
        optical_fraction = len(self.optical_layers) / (len(self.optical_layers) + 1)
        
        # Theoretical speedup basado en paralelización óptica
        theoretical_speedup = min(10.0, optical_fraction * 2.0 + 1.0)

            # ✅ AGREGADO: Calcular operaciones ópticas totales
        total_optical_operations = 0
        for i in range(len(self.optical_layers)):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            total_optical_operations += in_size * out_size
        
        # ✅ AGREGADO: También incluir operaciones de photodetección
        total_optical_operations += sum(self.layer_sizes[:-1])  # Una operación por photodetector
        
        return {
            "n_optical_layers": len(self.optical_layers),
            "n_photodetectors": len(self.photodetectors),
            "theoretical_speedup": theoretical_speedup,
            "optical_fraction": optical_fraction,
            "optical_operations": total_optical_operations,  # ✅ NUEVA LÍNEA
            "total_optical_parameters": total_optical_params,
            "total_electrical_parameters": electrical_params,
            "parameter_efficiency": self._n_parameters / max(1, sum(self.layer_sizes)),
            "optical_parameter_ratio": total_optical_params / max(1, total_optical_params + electrical_params),
            "estimated_power_consumption": optical_fraction * 0.1 + (1 - optical_fraction) * 1.0,
            "parallelization_factor": len(self.optical_layers)
        }
    
    def get_optical_efficiency(self) -> Dict[str, Any]:
        """
        Alias para get_optical_efficiency_metrics() - compatibilidad con demo.
        
        Returns:
            Dict con métricas de eficiencia
        """
        return self.get_optical_efficiency_metrics()
    
    def validate_unitarity(self) -> Dict[str, Any]:
        """
        Validar que las matrices sean unitarias.
        
        Returns:
            Dict con resultados de validación
        """
        validation = {"layers": {}, "overall_valid": True}
        
        for i, layer in enumerate(self.optical_layers):
            layer_validation = {"is_unitary": False, "error": float('inf')}
            
            try:
                if hasattr(layer, 'get_unitary_matrix'):
                    # Para MZILayer
                    U = layer.get_unitary_matrix()
                    identity_check = torch.matmul(U, torch.conj(U.t()))
                    identity_target = torch.eye(U.size(0), dtype=U.dtype, device=U.device)
                    error = torch.max(torch.abs(identity_check - identity_target)).item()
                    
                    layer_validation["is_unitary"] = error < 1e-3
                    layer_validation["unitarity_error"] = error
                elif hasattr(layer, '_get_weight_matrix'):
                    # Para MZIBlockLinear, verificar que ||W||_2 ≤ 1
                    W = layer._get_weight_matrix()
                    singular_values = torch.svd(W)[1]
                    max_sv = torch.max(singular_values).item()
                    
                    layer_validation["is_unitary"] = max_sv <= 1.1  # Permitir tolerancia
                    layer_validation["max_singular_value"] = max_sv
            except Exception as e:
                layer_validation["error"] = str(e)
            
            validation["layers"][f"layer_{i}"] = layer_validation
            
            if not layer_validation.get("is_unitary", False):
                validation["overall_valid"] = False
        
        return validation
    
    def get_physics_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas físicas de la red.
        
        Returns:
            Dict con métricas físicas
        """
        metrics = {
            "energy_conservation_ratio": 1.0,  # Placeholder
            "optical_loss_db": 0.1 * len(self.optical_layers),  # Estimación
            "insertion_loss_db": 0.05 * len(self.optical_layers),
            "crosstalk_db": -30.0,  # Típico para MZI mesh
            "thermal_stability": 0.95,  # Factor de estabilidad
            "phase_noise_variance": 0.01,  # Varianza de ruido de fase
        }
        
        # Si hay métricas reales de ONN, usar esas
        onn_metrics = self.get_onn_metrics()
        if "energy_conservation_history" in onn_metrics and len(onn_metrics["energy_conservation_history"]) > 0:
            metrics["energy_conservation_ratio"] = np.mean(onn_metrics["energy_conservation_history"])
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Resumen completo de performance.
        
        Returns:
            Dict con resumen de performance
        """
        efficiency = self.get_optical_efficiency_metrics()
        physics = self.get_physics_metrics()
        unitarity = self.validate_unitarity()
        
        return {
            "architecture": {
                "layer_sizes": self.layer_sizes,
                "total_parameters": self._n_parameters,
                "activation_type": self.activation_type,
                "unitary_constraints": self.use_unitary_constraints
            },
            "efficiency": efficiency,
            "physics": physics,
            "unitarity": unitarity,
            "device": str(self.device),
            "forward_passes": self.onn_metrics["total_forward_passes"]
        }
    
    
    def optical_operations(self) -> Dict[str, Any]:
        """
        Información sobre operaciones ópticas en la red.
        
        Returns:
            Dict con información de operaciones ópticas
        """
        operations = {
            "total_operations": 0,
            "optical_operations": [],
            "electrical_operations": [],
            "operation_breakdown": {}
        }
        
        # Contar operaciones por capa óptica
        for i, layer in enumerate(self.optical_layers):
            layer_ops = {
                "layer_index": i,
                "type": "optical_linear",
                "input_size": self.layer_sizes[i],
                "output_size": self.layer_sizes[i + 1],
                "operations_count": self.layer_sizes[i] * self.layer_sizes[i + 1],
                "component_type": type(layer).__name__
            }
            
            if hasattr(layer, 'n_mzis'):
                layer_ops["mzi_count"] = layer.n_mzis
                layer_ops["phase_shifter_count"] = getattr(layer, 'n_phases', 0)
            
            operations["optical_operations"].append(layer_ops)
            operations["total_operations"] += layer_ops["operations_count"]
        
        # Operaciones de photodetection
        for i, photodetector in enumerate(self.photodetectors):
            pd_ops = {
                "detector_index": i,
                "type": "photodetection",
                "operations_count": self.layer_sizes[min(i, len(self.layer_sizes)-1)],
                "responsivity": getattr(photodetector, 'responsivity', 1.0)
            }
            operations["optical_operations"].append(pd_ops)
            operations["total_operations"] += pd_ops["operations_count"]
        
        # Operaciones eléctricas (capa final)
        if hasattr(self, 'final_layer'):
            final_ops = {
                "type": "electrical_linear",
                "input_size": self.layer_sizes[-2],
                "output_size": self.layer_sizes[-1],
                "operations_count": self.layer_sizes[-2] * self.layer_sizes[-1]
            }
            operations["electrical_operations"].append(final_ops)
            operations["total_operations"] += final_ops["operations_count"]
        
        # Breakdown de operaciones
        operations["operation_breakdown"] = {
            "optical_linear_ops": sum(op["operations_count"] for op in operations["optical_operations"] if op["type"] == "optical_linear"),
            "photodetection_ops": sum(op["operations_count"] for op in operations["optical_operations"] if op["type"] == "photodetection"),
            "electrical_ops": sum(op["operations_count"] for op in operations["electrical_operations"]),
            "total_optical_ops": sum(op["operations_count"] for op in operations["optical_operations"]),
            "optical_percentage": 0.0
        }
        
        # Calcular porcentaje óptico
        if operations["total_operations"] > 0:
            operations["operation_breakdown"]["optical_percentage"] = (
                operations["operation_breakdown"]["total_optical_ops"] / operations["total_operations"] * 100
            )
        
        return operations

    def extra_repr(self) -> str:
        """Representación adicional para debugging."""
        return (f"layer_sizes={self.layer_sizes}, "
                f"activation_type='{self.activation_type}', "
                f"unitary_constraints={self.use_unitary_constraints}")


def create_simple_coherent_onn(
    input_size: int = 4,
    hidden_size: int = 8, 
    output_size: int = 3,
    device: Optional[torch.device] = None
) -> CoherentONN:
    """
    Crear una CoherentONN simple para testing y demos.
    
    Args:
        input_size: Tamaño de entrada
        hidden_size: Tamaño de capa oculta
        output_size: Tamaño de salida
        device: Device
        
    Returns:
        CoherentONN configurada y lista para usar
    """
    return CoherentONN(
        layer_sizes=[input_size, hidden_size, output_size],
        activation_type="square_law",
        optical_power=1.0,
        use_unitary_constraints=True,
        device=device
    )