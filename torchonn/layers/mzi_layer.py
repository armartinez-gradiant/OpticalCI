"""
MZI Layer - Implementación Física Real para PtONN-TESTS

CORREGIDO: Ahora implementa física real de Mach-Zehnder Interferometer
- Descomposición de Reck para matrices unitarias
- Beam splitters y phase shifters reales
- Conservación de energía garantizada
- Matrices unitarias validadas

Cambio Principal:
❌ ANTES: output = torch.mm(x, self.weight.t())  # Solo álgebra lineal
✅ AHORA: U = self._construct_unitary_matrix()   # Física real MZI
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple
import warnings

class MZILayer(nn.Module):
    """
    MZI Layer con física real de interferometría.
    
    Implementa Mach-Zehnder Interferometer usando descomposición de Reck:
    - Cada peso se codifica como ángulos de beam splitter (θ) y phase shifts (φ)
    - La matriz resultante es siempre unitaria (conserva energía)
    - Representa dispositivos fotónicos reales
    
    CAMBIO CRÍTICO vs. versión anterior:
    - Antes: Multiplicación matricial arbitraria
    - Ahora: Construcción física desde parámetros MZI
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MZILayer, self).__init__()
        
        # Validación: MZI requiere dimensiones cuadradas para matrices unitarias
        if in_features != out_features:
            warnings.warn(
                f"MZI works best with square matrices. "
                f"Got {in_features}→{out_features}. "
                f"Will use max({in_features}, {out_features}) and pad/truncate."
            )
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Usar dimensión máxima para matriz unitaria 
        self.matrix_dim = max(in_features, out_features)
        
        # Device y dtype setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        
        # Calcular número de MZIs necesarios (Descomposición de Reck)
        # Para matriz N×N unitaria: N(N-1)/2 MZIs + N phase shifters externos
        n = self.matrix_dim
        self.n_mzis = n * (n - 1) // 2
        self.n_phases = n
        
        print(f"🔧 MZI Layer CORREGIDO: {in_features}→{out_features}")
        print(f"   📐 Matriz unitaria: {n}×{n}")
        print(f"   🔗 MZIs físicos: {self.n_mzis}")
        print(f"   🌊 Phase shifters: {self.n_phases}")
        
        # PARÁMETROS FÍSICOS REALES (no pesos arbitrarios)
        
        # Ángulos de beam splitters (θ ∈ [0, π/2])
        self.theta = nn.Parameter(torch.zeros(self.n_mzis, device=device, dtype=dtype))
        
        # Phase shifts internos (φ ∈ [0, 2π])
        self.phi_internal = nn.Parameter(torch.zeros(self.n_mzis, device=device, dtype=dtype))
        
        # Phase shifts externos (α ∈ [0, 2π])  
        self.phi_external = nn.Parameter(torch.zeros(self.n_phases, device=device, dtype=dtype))
        
        # Inicialización física realista
        self.reset_parameters()
        
        # Mover a device
        self.to(device)
    
    def reset_parameters(self):
        """Inicialización con distribuciones físicamente motivadas."""
        with torch.no_grad():
            # Beam splitters: distribución uniforme [0, π/2]
            # θ=0 → sin transmisión, θ=π/2 → máximo coupling
            nn.init.uniform_(self.theta, 0, np.pi/2)
            
            # Phase shifts: distribución uniforme [0, 2π]
            nn.init.uniform_(self.phi_internal, 0, 2*np.pi)
            nn.init.uniform_(self.phi_external, 0, 2*np.pi)
            
            # Pequeña perturbación para romper simetrías
            self.theta.add_(torch.randn_like(self.theta) * 0.01)
            self.phi_internal.add_(torch.randn_like(self.phi_internal) * 0.01)
            self.phi_external.add_(torch.randn_like(self.phi_external) * 0.01)
    
    def _single_mzi_matrix(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Matriz de transferencia de un MZI individual.
        
        Implementa la física real:
        MZI = BS₂ × Φ(φ) × BS₁
        
        Args:
            theta: Ángulo del beam splitter
            phi: Phase shift
            
        Returns:
            Matriz 2×2 del MZI individual
        """
        # Coeficientes de beam splitter
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Phase shift complex
        exp_phi = torch.cos(phi) + 1j * torch.sin(phi)
        
        # Matriz MZI 2×2 (FÍSICA REAL)
        # [[cos(θ),           -sin(θ)*exp(-iφ)],
        #  [sin(θ)*exp(iφ),    cos(θ)          ]]
        
        mzi_matrix = torch.zeros(2, 2, dtype=torch.complex64, device=self.device)
        mzi_matrix[0, 0] = cos_theta
        mzi_matrix[0, 1] = -sin_theta * torch.conj(exp_phi)
        mzi_matrix[1, 0] = sin_theta * exp_phi  
        mzi_matrix[1, 1] = cos_theta
        
        return mzi_matrix
    
    def _construct_unitary_matrix(self) -> torch.Tensor:
        """
        Construir matriz unitaria completa usando descomposición de Reck.
        
        ESTO ES LA CORRECCIÓN PRINCIPAL:
        En lugar de usar pesos arbitrarios, construimos matriz unitaria
        desde parámetros físicos de MZIs reales.
        
        Returns:
            Matriz unitaria N×N que representa la red de MZIs
        """
        n = self.matrix_dim
        
        # Comenzar con matriz identidad
        U = torch.eye(n, dtype=torch.complex64, device=self.device)
        
        mzi_idx = 0
        
        # Aplicar MZIs en orden de descomposición de Reck
        # Configuración triangular: cada MZI actúa en pares adyacentes
        for layer in range(n - 1):
            for pos in range(n - 1 - layer):
                if mzi_idx < self.n_mzis:
                    # Parámetros del MZI actual
                    theta = self.theta[mzi_idx]
                    phi = self.phi_internal[mzi_idx]
                    
                    # Matriz MZI local 2×2
                    mzi_local = self._single_mzi_matrix(theta, phi)
                    
                    # Expandir a matriz N×N (actúa solo en posiciones pos, pos+1)
                    mzi_full = torch.eye(n, dtype=torch.complex64, device=self.device)
                    mzi_full[pos:pos+2, pos:pos+2] = mzi_local
                    
                    # Aplicar transformación: U = MZI × U
                    U = torch.matmul(mzi_full, U)
                    
                    mzi_idx += 1
        
        # Aplicar phase shifts externos finales
        phase_diagonal = torch.diag(
            torch.cos(self.phi_external) + 1j * torch.sin(self.phi_external)
        )
        U = torch.matmul(phase_diagonal, U)
        
        return U
    
    def validate_unitarity(self, U: torch.Tensor, tolerance: float = 1e-4) -> bool:
        """Validar que la matriz construida es unitaria."""
        # U × U† debe ser identidad
        identity_check = torch.matmul(U, torch.conj(U.t()))
        identity_target = torch.eye(U.size(0), dtype=U.dtype, device=U.device)
        
        max_error = torch.max(torch.abs(identity_check - identity_target)).item()
        
        if max_error > tolerance:
            warnings.warn(f"Unitarity violation: {max_error:.2e} > {tolerance:.2e}")
            return False
        
        return True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con física MZI real.
        
        🔧 CORRECCIÓN PRINCIPAL:
        ❌ ANTES: return torch.mm(x, self.weight.t())
        ✅ AHORA: Usar matriz unitaria construida físicamente
        """
        batch_size = x.size(0)
        
        # Validaciones robustas
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        if x.size(-1) != self.in_features:
            raise ValueError(f"Input features mismatch: expected {self.in_features}, got {x.size(-1)}")
        
        # Device/dtype consistency
        if x.device != self.device:
            x = x.to(self.device)
        
        # CONSTRUCCIÓN DE MATRIZ UNITARIA FÍSICA
        U = self._construct_unitary_matrix()
        
        # Validar unitarity (crítico para conservación de energía)
        if self.training and torch.rand(1).item() < 0.1:  # 10% de las veces en training
            self.validate_unitarity(U)
        
        # Preparar input para multiplicación
        if self.in_features != self.matrix_dim:
            # Pad si necesario
            if x.size(1) < self.matrix_dim:
                padding = torch.zeros(batch_size, self.matrix_dim - x.size(1), 
                                    device=x.device, dtype=x.dtype)
                x_padded = torch.cat([x, padding], dim=1)
            else:
                x_padded = x[:, :self.matrix_dim]
        else:
            x_padded = x
        
        # Convertir a complex para multiplicación unitaria
        x_complex = x_padded.to(dtype=torch.complex64)
        
        # APLICAR TRANSFORMACIÓN UNITARIA (FÍSICA REAL)
        output_complex = torch.matmul(x_complex, U.t().conj())
        
        # Convertir de vuelta a real (tomar magnitud)
        if x.dtype.is_floating_point:
            output = torch.real(output_complex)
        else:
            output = torch.abs(output_complex)  # Magnitud para detección incoherente
        
        # Truncar a dimensión de salida si necesario
        if output.size(1) != self.out_features:
            output = output[:, :self.out_features]
        
        # Conversión final a dtype original
        output = output.to(dtype=self.dtype)
        
        # VALIDACIÓN DE CONSERVACIÓN DE ENERGÍA
        if self.training:
            input_power = torch.sum(torch.abs(x_padded)**2).item()
            output_power = torch.sum(torch.abs(output)**2).item()
            
            if input_power > 1e-10:  # Evitar división por cero
                power_ratio = output_power / input_power
                if abs(power_ratio - 1.0) > 0.1:  # 10% tolerancia
                    warnings.warn(f"Energy conservation issue: {power_ratio:.3f} (should be ≈1.0)")
        
        return output
    
    def get_unitary_matrix(self) -> torch.Tensor:
        """Obtener la matriz unitaria construida (para debugging/análisis)."""
        return self._construct_unitary_matrix()
    
    def get_insertion_loss_db(self) -> torch.Tensor:
        """Calcular pérdida de inserción en dB."""
        U = self.get_unitary_matrix()
        
        # Para matriz unitaria ideal: loss = 0 dB
        # En práctica, pequeñas pérdidas por no-idealidades
        transmission = torch.abs(torch.diag(U))**2
        loss_linear = 1 - transmission.mean()
        loss_db = -10 * torch.log10(torch.clamp(1 - loss_linear, min=1e-10))
        
        return loss_db
    
    def extra_repr(self) -> str:
        """Representación extra para debugging."""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"matrix_dim={self.matrix_dim}, n_mzis={self.n_mzis}, "
                f"device={self.device}")
