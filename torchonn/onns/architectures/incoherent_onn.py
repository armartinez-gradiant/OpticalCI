"""
Incoherent Optical Neural Network (IncoherentONN) - IMPLEMENTACIÓN COMPLETA

📚 Based on: 
- Tait et al. "Neuromorphic photonic networks using silicon photonic weight banks" (2017)
- Hughes et al. "Training of photonic neural networks through in situ backpropagation" (2018)
- Feldmann et al. "Parallel convolutional processing using an integrated photonic tensor core" (2021)

🔬 Arquitectura: Intensity-based processing usando microring weight banks + WDM
🎯 Ventajas: Escalabilidad, robustez al ruido de fase, fabrication tolerance
⚡ Operación: y = Σᵢ wᵢ|xᵢ|² donde wᵢ = transmisión del microring i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
import warnings

# Imports de componentes OpticalCI existentes
from ...layers import MicroringResonator, Photodetector
from ...components import WDMMultiplexer, MRRWeightBank
from .base_onn import BaseONN


class IncoherentLayer(nn.Module):
    """
    Capa fotónica incoherente usando microring weight bank.
    
    Implementa: y = Σλ Σᵢ wᵢλ|xᵢ|²
    donde wᵢλ = transmisión del microring i en wavelength λ
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_wavelengths: int = 4,
        wavelength_range: Tuple[float, float] = (1530e-9, 1570e-9),
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_wavelengths = n_wavelengths
        
        # Crear wavelengths para WDM
        wl_min, wl_max = wavelength_range
        self.wavelengths = torch.linspace(wl_min, wl_max, n_wavelengths, device=device)
        
        # Sistema WDM
        self.wdm_system = WDMMultiplexer(
            wavelengths=self.wavelengths.cpu().numpy().tolist(),
            device=device
        )
        
        # MRR Weight Bank (matriz de microrings)
        self.weight_bank = MRRWeightBank(
            n_inputs=in_features,
            n_outputs=out_features, 
            n_wavelengths=n_wavelengths,
            wavelength_range=wavelength_range,
            device=device
        )
        
        # Photodetectors para cada output
        self.photodetectors = nn.ModuleList([
            Photodetector(responsivity=1.0, device=device) 
            for _ in range(out_features)
        ])
        
        print(f"🔗 IncoherentLayer: {in_features}→{out_features}, {n_wavelengths} wavelengths")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass incoherente.
        
        Args:
            x: Input [batch_size, in_features]
            
        Returns:
            Output [batch_size, out_features]
        """
        batch_size = x.size(0)
        
        # 1. Convertir a intensidad |x|²
        intensity = torch.abs(x)**2
        
        # 2. Expandir para WDM channels: [batch_size, in_features, n_wavelengths]
        intensity_wdm = intensity.unsqueeze(2).expand(-1, -1, self.n_wavelengths)
        
        # 3. Aplicar weight bank (microring array processing)
        weighted_signals = self.weight_bank(intensity_wdm)
        
        # 4. Photodetection para cada output
        outputs = []
        for i in range(self.out_features):
            # Extraer señal para output i
            output_signal = weighted_signals[:, i, :]  # [batch_size, n_wavelengths]
            
            # Photodetection + suma sobre wavelengths
            detected = self.photodetectors[i](output_signal)
            summed = torch.sum(detected, dim=1, keepdim=True)  # [batch_size, 1]
            outputs.append(summed)
        
        # 5. Concatenar outputs
        result = torch.cat(outputs, dim=1)  # [batch_size, out_features]
        
        return result
    
    def get_weight_matrix(self) -> torch.Tensor:
        """Obtener matriz de pesos efectiva."""
        return self.weight_bank.get_weight_matrix()
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Métricas de eficiencia de la capa."""
        return {
            "microring_count": self.in_features * self.out_features,
            "wavelength_channels": self.n_wavelengths,
            "photodetector_count": self.out_features,
            "parallel_operations": self.in_features * self.out_features * self.n_wavelengths
        }


class IncoherentONN(BaseONN):
    """
    Incoherent Optical Neural Network - Arquitectura completa.
    
    Implementa red neuronal fotónica basada en intensidad usando:
    - Arrays de microrings para pesos
    - WDM para paralelización
    - Photodetección para conversión O-E
    - Activaciones eléctricas
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        n_wavelengths: int = 4,
        wavelength_range: Tuple[float, float] = (1530e-9, 1570e-9),
        activation_type: str = "relu",
        optical_power: float = 1.0,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Inicializar IncoherentONN.
        
        Args:
            layer_sizes: Lista de tamaños [input, hidden1, ..., output]
            n_wavelengths: Número de canales WDM
            wavelength_range: Rango de wavelengths (min, max)
            activation_type: Tipo de activación ("relu", "sigmoid", "tanh")
            optical_power: Potencia óptica normalizada
            device: Device para cálculo
        """
        # Validaciones
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes debe tener al menos 2 elementos")
        if n_wavelengths < 1:
            raise ValueError("n_wavelengths debe ser >= 1")
        
        # Inicializar clase base
        super().__init__(
            device=device,
            optical_power=optical_power,
            wavelength_channels=n_wavelengths,
            enable_physics_validation=True
        )
        
        # Configuración
        self.layer_sizes = layer_sizes
        self.n_wavelengths = n_wavelengths
        self.wavelength_range = wavelength_range
        self.activation_type = activation_type
        self.n_layers = len(layer_sizes) - 1
        
        # Crear arquitectura
        self._build_incoherent_layers()
        
        # Activación eléctrica
        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_type == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Activation type '{activation_type}' not supported")
        
        # Mover a device
        self.to(self.device)
        
        # Contar parámetros
        self._n_parameters = self._count_parameters()
        
        print(f"🔗 IncoherentONN initialized:")
        print(f"   Layer sizes: {layer_sizes}")
        print(f"   Wavelengths: {n_wavelengths}")
        print(f"   Parameters: ~{self._n_parameters:,}")
        print(f"   Activation: {activation_type}")
        print(f"   Device: {device}")
    
    def _build_incoherent_layers(self):
        """Construir capas fotónicas incoherentes."""
        self.incoherent_layers = nn.ModuleList()
        
        # Crear capas fotónicas (todas excepto la última)
        for i in range(self.n_layers - 1):
            layer = IncoherentLayer(
                in_features=self.layer_sizes[i],
                out_features=self.layer_sizes[i + 1],
                n_wavelengths=self.n_wavelengths,
                wavelength_range=self.wavelength_range,
                device=self.device
            )
            self.incoherent_layers.append(layer)
        
        # Capa final eléctrica
        self.final_layer = nn.Linear(
            self.layer_sizes[-2], 
            self.layer_sizes[-1], 
            device=self.device
        )
    
    def _count_parameters(self) -> int:
        """Contar parámetros aproximados."""
        total = 0
        
        # Capas fotónicas: microrings + WDM
        for i in range(len(self.layer_sizes) - 2):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            # Cada microring tiene ~2 parámetros (resonance + coupling)
            total += in_size * out_size * self.n_wavelengths * 2
        
        # Capa final eléctrica
        total += self.layer_sizes[-2] * self.layer_sizes[-1]
        
        return total
    
    def _forward_optical(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass óptico (requerido por BaseONN)."""
        return self.forward(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo de IncoherentONN.
        
        Args:
            x: Input [batch_size, input_features]
            
        Returns:
            Output [batch_size, output_features]
        """
        # Validaciones
        if x.dim() != 2:
            raise ValueError(f"Input debe ser 2D, recibido: {x.dim()}D")
        if x.size(1) != self.layer_sizes[0]:
            raise ValueError(f"Input size {x.size(1)} != expected {self.layer_sizes[0]}")
        
        x = x.to(self.device)
        
        # Normalización de entrada
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        x = x * torch.sqrt(torch.tensor(self.optical_power, device=self.device))
        
        # Forward a través de capas fotónicas
        current = x
        for i, layer in enumerate(self.incoherent_layers):
            # Capa fotónica incoherente
            current = layer(current)
            
            # Activación eléctrica
            current = self.activation(current)
            
            # Re-normalización para mantener potencia óptica
            if i < len(self.incoherent_layers) - 1:
                current = current / (torch.norm(current, dim=1, keepdim=True) + 1e-8)
                current = current * torch.sqrt(torch.tensor(self.optical_power, device=self.device))
        
        # Capa final eléctrica
        output = self.final_layer(current)
        
        # Actualizar métricas
        self.onn_metrics["total_forward_passes"] += 1
        
        return output
    
    def get_optical_efficiency_metrics(self) -> Dict[str, Any]:
        """Métricas de eficiencia óptica específicas para IncoherentONN."""
        # Contar componentes ópticos
        total_microrings = 0
        total_photodetectors = 0
        
        for layer in self.incoherent_layers:
            metrics = layer.get_efficiency_metrics()
            total_microrings += metrics["microring_count"]
            total_photodetectors += metrics["photodetector_count"]
        
        # Operaciones paralelas por WDM
        parallel_ops = total_microrings * self.n_wavelengths
        
        # Fracción óptica
        optical_layers = len(self.incoherent_layers)
        total_layers = optical_layers + 1  # +1 para capa final eléctrica
        optical_fraction = optical_layers / total_layers
        
        return {
            "architecture_type": "incoherent",
            "total_microrings": total_microrings,
            "total_photodetectors": total_photodetectors,
            "wavelength_channels": self.n_wavelengths,
            "parallel_operations": parallel_ops,
            "optical_fraction": optical_fraction,
            "optical_operations": parallel_ops,  # Para compatibilidad con tests
            "theoretical_speedup": min(20.0, self.n_wavelengths * optical_fraction + 1.0),
            "scalability_factor": self.n_wavelengths,
            "power_efficiency": optical_fraction * self.n_wavelengths / 10.0,  # Estimación
            "fabrication_tolerance": 0.8,  # Mayor tolerancia que coherent
            "phase_sensitivity": 0.1  # Muy baja (ventaja incoherente)
        }
    
    def get_optical_efficiency(self) -> Dict[str, Any]:
        """Alias para compatibilidad."""
        return self.get_optical_efficiency_metrics()
    
    def validate_physics(self) -> Dict[str, Any]:
        """Validar física específica de IncoherentONN."""
        validation = {}
        
        # 1. Validar rangos de transmisión de microrings
        valid_transmissions = True
        min_transmission = 1.0
        max_transmission = 0.0
        
        for layer in self.incoherent_layers:
            try:
                weight_matrix = layer.get_weight_matrix()  # [out, in, wavelengths]
                
                layer_min = torch.min(weight_matrix).item()
                layer_max = torch.max(weight_matrix).item()
                
                min_transmission = min(min_transmission, layer_min)
                max_transmission = max(max_transmission, layer_max)
                
                # Transmisiones deben estar en [0, 1]
                if layer_min < -0.01 or layer_max > 1.01:
                    valid_transmissions = False
                    
            except Exception as e:
                valid_transmissions = False
        
        validation["valid_transmissions"] = valid_transmissions
        validation["transmission_range"] = (min_transmission, max_transmission)
        
        # 2. Conservación de energía (más relajada para incoherent)
        validation["energy_conservation_type"] = "intensity_based"
        validation["allows_energy_loss"] = True  # Los microrings pueden absorber
        
        # 3. WDM channel separation
        if len(self.wavelength_range) == 2:
            wl_span = self.wavelength_range[1] - self.wavelength_range[0]
            channel_spacing = wl_span / max(1, self.n_wavelengths - 1)
            
            validation["wdm_channel_spacing_nm"] = channel_spacing * 1e9
            validation["wdm_spacing_adequate"] = channel_spacing > 1e-12  # > 1 pm
        
        return validation
    
    def get_architecture_comparison(self) -> Dict[str, Any]:
        """Comparación con CoherentONN."""
        return {
            "architecture": "IncoherentONN",
            "operation_principle": "intensity_based",
            "components": ["microring_arrays", "wdm_multiplexers", "photodetectors"],
            "advantages": [
                "scalable_with_wdm",
                "phase_insensitive", 
                "fabrication_tolerant",
                "power_efficient"
            ],
            "limitations": [
                "positive_weights_only",  # Inicialmente
                "quantized_precision",
                "wdm_latency"
            ],
            "vs_coherent": {
                "robustness": "higher",
                "scalability": "better", 
                "precision": "lower",
                "power": "lower"
            }
        }
    
    def extra_repr(self) -> str:
        """Representación para debugging."""
        return (f"layer_sizes={self.layer_sizes}, "
                f"n_wavelengths={self.n_wavelengths}, "
                f"activation='{self.activation_type}'")


def create_simple_incoherent_onn(
    input_size: int = 4,
    hidden_size: int = 8,
    output_size: int = 3,
    n_wavelengths: int = 4,
    device: Optional[torch.device] = None
) -> IncoherentONN:
    """
    Crear IncoherentONN simple para testing.
    
    Args:
        input_size: Tamaño de entrada
        hidden_size: Tamaño de capa oculta
        output_size: Tamaño de salida  
        n_wavelengths: Canales WDM
        device: Device
        
    Returns:
        IncoherentONN configurada
    """
    return IncoherentONN(
        layer_sizes=[input_size, hidden_size, output_size],
        n_wavelengths=n_wavelengths,
        activation_type="relu",
        optical_power=1.0,
        device=device
    )


if __name__ == "__main__":
    # Test básico
    print("🧪 Testing IncoherentONN...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear red simple
    onn = create_simple_incoherent_onn(device=device)
    
    # Test forward
    x = torch.randn(4, 4, device=device) * 0.5
    y = onn(x)
    
    print(f"✅ Forward pass: {x.shape} → {y.shape}")
    print(f"📊 Efficiency: {onn.get_optical_efficiency_metrics()['optical_fraction']:.2f}")
    print(f"🔬 Physics valid: {onn.validate_physics()['valid_transmissions']}")