"""
Tests para MZI Layers - PtONN-TESTS

Suite completa de tests que valida:
- Comportamiento unitario de matrices
- Conservación de energía
- Diferentes modos de operación (USV, weight, phase)
- Gradientes y backpropagation
- Edge cases y robustez
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

# Import del módulo a testear
from torchonn.layers import MZILayer, MZIBlockLinear


class TestMZILayer:
    """Tests para MZILayer con validación física."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def mzi_4x4(self, device):
        """Fixture para MZI 4x4 estándar."""
        return MZILayer(in_features=4, out_features=4, device=device)
    
    @pytest.fixture
    def input_batch(self, device):
        """Fixture para batch de entrada."""
        return torch.randn(16, 4, device=device, dtype=torch.float32)
    
    def test_mzi_initialization(self, device):
        """Test: Inicialización correcta del MZI."""
        mzi = MZILayer(in_features=4, out_features=4, device=device)
        
        # Verificar parámetros básicos
        assert mzi.in_features == 4
        assert mzi.out_features == 4
        assert mzi.matrix_dim == 4
        assert mzi.device == device
        
        # Verificar que se crearon los parámetros necesarios
        assert hasattr(mzi, 'theta')
        assert hasattr(mzi, 'phi_internal')
        assert hasattr(mzi, 'phi_external')
        
        # Verificar dimensiones de parámetros
        expected_n_mzis = 4 * (4 - 1) // 2  # 6 MZIs para 4x4
        assert mzi.theta.shape == (expected_n_mzis,)
        assert mzi.phi_internal.shape == (expected_n_mzis,)
        assert mzi.phi_external.shape == (4,)
    
    def test_unitary_matrix_property(self, mzi_4x4):
        """Test: La matriz construida es unitaria."""
        U = mzi_4x4.get_unitary_matrix()
        
        # Test 1: Dimensiones correctas
        assert U.shape == (4, 4)
        assert U.dtype == torch.complex64
        
        # Test 2: Propiedad unitaria U @ U† = I
        identity_check = torch.matmul(U, torch.conj(U.t()))
        identity_target = torch.eye(4, dtype=torch.complex64, device=mzi_4x4.device)
        
        max_error = torch.max(torch.abs(identity_check - identity_target))
        assert max_error < 1e-5, f"Unitarity violation: {max_error:.2e}"
        
        # Test 3: Determinante = ±1 (propiedad de matrices unitarias)
        det = torch.det(U)
        assert abs(torch.abs(det) - 1.0) < 1e-5, f"Determinant not ±1: {det}"
    
    def test_energy_conservation(self, mzi_4x4, input_batch):
        """Test: Conservación de energía perfecta."""
        output = mzi_4x4(input_batch)
        
        # Calcular energía de entrada y salida
        input_energy = torch.sum(torch.abs(input_batch)**2, dim=1)
        output_energy = torch.sum(torch.abs(output)**2, dim=1)
        
        # Test: Conservación de energía perfecta (matriz unitaria)
        energy_ratio = output_energy / torch.clamp(input_energy, min=1e-10)
        energy_conservation = torch.mean(energy_ratio)
        energy_std = torch.std(energy_ratio)
        
        assert abs(energy_conservation - 1.0) < 1e-4, f"Energy not conserved: {energy_conservation:.6f}"
        assert energy_std < 1e-4, f"Energy conservation unstable: std = {energy_std:.6f}"
    
    def test_different_dimensions(self, device):
        """Test: Diferentes dimensiones de MZI."""
        dimensions = [(2, 2), (3, 3), (4, 4), (8, 8)]
        
        for in_dim, out_dim in dimensions:
            mzi = MZILayer(in_features=in_dim, out_features=out_dim, device=device)
            
            # Test forward pass
            input_tensor = torch.randn(4, in_dim, device=device, dtype=torch.float32)
            output = mzi(input_tensor)
            
            # Verificar dimensiones
            assert output.shape == (4, out_dim)
            
            # Verificar conservación de energía para matrices cuadradas
            if in_dim == out_dim:
                input_energy = torch.sum(torch.abs(input_tensor)**2, dim=1)
                output_energy = torch.sum(torch.abs(output)**2, dim=1)
                energy_ratio = torch.mean(output_energy / torch.clamp(input_energy, min=1e-10))
                
                assert abs(energy_ratio - 1.0) < 1e-3, f"Energy not conserved for {in_dim}x{out_dim}"
    
    def test_non_square_matrices(self, device):
        """Test: Matrices no cuadradas (con warning)."""
        # Test 1: más salidas que entradas
        mzi_expand = MZILayer(in_features=3, out_features=5, device=device)
        input_3d = torch.randn(4, 3, device=device, dtype=torch.float32)
        output_5d = mzi_expand(input_3d)
        
        assert output_5d.shape == (4, 5)
        
        # Test 2: menos salidas que entradas  
        mzi_reduce = MZILayer(in_features=5, out_features=3, device=device)
        input_5d = torch.randn(4, 5, device=device, dtype=torch.float32)
        output_3d = mzi_reduce(input_5d)
        
        assert output_3d.shape == (4, 3)
    
    def test_gradients_flow(self, device):
        """Test: Los gradientes fluyen correctamente."""
        mzi = MZILayer(in_features=4, out_features=4, device=device)
        input_tensor = torch.randn(8, 4, device=device, dtype=torch.float32, requires_grad=True)
        
        # Forward pass
        output = mzi(input_tensor)
        loss = torch.mean(output**2)
        
        # Backward pass
        loss.backward()
        
        # Test: Gradientes en input
        assert input_tensor.grad is not None, "No gradients on input"
        assert not torch.all(input_tensor.grad == 0), "Input gradients are zero"
        
        # Test: Gradientes en parámetros
        assert mzi.theta.grad is not None, "No gradients on theta"
        assert mzi.phi_internal.grad is not None, "No gradients on phi_internal"
        assert mzi.phi_external.grad is not None, "No gradients on phi_external"
    
    def test_insertion_loss(self, mzi_4x4):
        """Test: Pérdida de inserción (debe ser ~0 para MZI ideal)."""
        insertion_loss = mzi_4x4.get_insertion_loss_db()
        
        # Para MZI unitario ideal, pérdida debe ser muy baja
        assert insertion_loss < 1.0, f"Insertion loss too high: {insertion_loss:.3f} dB"
        assert insertion_loss > -1.0, f"Insertion loss suspiciously low: {insertion_loss:.3f} dB"
    
    def test_parameter_reset(self, device):
        """Test: Reset de parámetros funciona."""
        mzi = MZILayer(in_features=4, out_features=4, device=device)
        
        # Guardar parámetros iniciales
        theta_initial = mzi.theta.clone()
        phi_int_initial = mzi.phi_internal.clone()
        phi_ext_initial = mzi.phi_external.clone()
        
        # Modificar parámetros
        with torch.no_grad():
            mzi.theta.fill_(1.0)
            mzi.phi_internal.fill_(1.0)
            mzi.phi_external.fill_(1.0)
        
        # Reset
        mzi.reset_parameters()
        
        # Verificar que cambiaron
        assert not torch.allclose(mzi.theta, theta_initial), "Theta not reset"
        assert not torch.allclose(mzi.phi_internal, phi_int_initial), "Phi internal not reset"
        assert not torch.allclose(mzi.phi_external, phi_ext_initial), "Phi external not reset"


class TestMZIBlockLinear:
    """Tests para MZIBlockLinear con diferentes modos."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_usv_mode(self, device):
        """Test: Modo USV funciona correctamente."""
        mzi = MZIBlockLinear(
            in_features=6,
            out_features=4,
            mode="usv",
            device=device
        )
        
        # Test forward pass
        input_tensor = torch.randn(8, 6, device=device, dtype=torch.float32)
        output = mzi(input_tensor)
        
        assert output.shape == (8, 4)
        
        # Test que los parámetros USV existen
        assert hasattr(mzi, 'u_matrix')
        assert hasattr(mzi, 's_matrix')
        assert hasattr(mzi, 'v_matrix')
        
        # Test dimensiones USV
        assert mzi.u_matrix.shape == (4, 4)
        assert mzi.v_matrix.shape == (6, 6)
        assert mzi.s_matrix.shape == (4,)  # min(4, 6) = 4
    
    def test_weight_mode(self, device):
        """Test: Modo weight funciona correctamente."""
        mzi = MZIBlockLinear(
            in_features=5,
            out_features=3,
            mode="weight",
            device=device
        )
        
        # Test forward pass
        input_tensor = torch.randn(4, 5, device=device, dtype=torch.float32)
        output = mzi(input_tensor)
        
        assert output.shape == (4, 3)
        
        # Test que weight existe
        assert hasattr(mzi, 'weight')
        assert mzi.weight.shape == (3, 5)
    
    def test_phase_mode(self, device):
        """Test: Modo phase funciona correctamente."""
        mzi = MZIBlockLinear(
            in_features=4,
            out_features=4,
            mode="phase",
            device=device
        )
        
        # Test forward pass
        input_tensor = torch.randn(6, 4, device=device, dtype=torch.float32)
        output = mzi(input_tensor)
        
        assert output.shape == (6, 4)
        
        # Test que phases existe
        assert hasattr(mzi, 'phases')
        assert mzi.phases.shape == (8,)  # 4 + 4 = 8
    
    def test_mode_consistency(self, device):
        """Test: Todos los modos producen outputs consistentes."""
        in_features, out_features = 4, 4
        input_tensor = torch.randn(5, in_features, device=device, dtype=torch.float32)
        
        modes = ["usv", "weight", "phase"]
        outputs = {}
        
        for mode in modes:
            mzi = MZIBlockLinear(
                in_features=in_features,
                out_features=out_features,
                mode=mode,
                device=device
            )
            outputs[mode] = mzi(input_tensor)
            
            # Test shape consistency
            assert outputs[mode].shape == (5, out_features), f"Wrong shape for mode {mode}"
            
            # Test no NaN/Inf
            assert not torch.any(torch.isnan(outputs[mode])), f"NaN in mode {mode}"
            assert not torch.any(torch.isinf(outputs[mode])), f"Inf in mode {mode}"
    
    def test_weight_matrix_construction(self, device):
        """Test: Construcción de matriz de pesos funciona."""
        modes = ["usv", "weight", "phase"]
        
        for mode in modes:
            mzi = MZIBlockLinear(
                in_features=3,
                out_features=2,
                mode=mode,
                device=device
            )
            
            # Obtener matriz de pesos
            weight_matrix = mzi._get_weight_matrix()
            
            # Test dimensiones
            assert weight_matrix.shape == (2, 3), f"Wrong weight shape for mode {mode}"
            
            # Test no NaN/Inf
            assert not torch.any(torch.isnan(weight_matrix)), f"NaN in weight matrix for mode {mode}"
            assert not torch.any(torch.isinf(weight_matrix)), f"Inf in weight matrix for mode {mode}"
    
    def test_gradients_all_modes(self, device):
        """Test: Gradientes funcionan en todos los modos."""
        modes = ["usv", "weight", "phase"]
        
        for mode in modes:
            mzi = MZIBlockLinear(
                in_features=4,
                out_features=3,
                mode=mode,
                device=device
            )
            
            input_tensor = torch.randn(4, 4, device=device, dtype=torch.float32, requires_grad=True)
            
            # Forward + backward
            output = mzi(input_tensor)
            loss = torch.mean(output**2)
            loss.backward()
            
            # Test gradients exist
            assert input_tensor.grad is not None, f"No input gradients for mode {mode}"
            
            # Test parameter gradients exist
            if mode == "usv":
                assert mzi.u_matrix.grad is not None, f"No U gradients for USV mode"
                assert mzi.s_matrix.grad is not None, f"No S gradients for USV mode"
                assert mzi.v_matrix.grad is not None, f"No V gradients for USV mode"
            elif mode == "weight":
                assert mzi.weight.grad is not None, f"No weight gradients for weight mode"
            elif mode == "phase":
                assert mzi.phases.grad is not None, f"No phase gradients for phase mode"


class TestMZIEdgeCases:
    """Tests de edge cases y robustez."""
    
    def test_single_input_output(self, device):
        """Test: MZI 1x1 (caso trivial)."""
        # Aunque físicamente no tiene sentido, debe funcionar matemáticamente
        mzi = MZILayer(in_features=1, out_features=1, device=device)
        input_tensor = torch.randn(3, 1, device=device, dtype=torch.float32)
        
        output = mzi(input_tensor)
        assert output.shape == (3, 1)
    
    def test_large_batch_size(self, device):
        """Test: Batch size grande."""
        mzi = MZILayer(in_features=4, out_features=4, device=device)
        large_batch = torch.randn(1000, 4, device=device, dtype=torch.float32)
        
        output = mzi(large_batch)
        assert output.shape == (1000, 4)
        
        # Test conservación de energía con batch grande
        input_energy = torch.sum(torch.abs(large_batch)**2, dim=1)
        output_energy = torch.sum(torch.abs(output)**2, dim=1)
        energy_ratio = torch.mean(output_energy / torch.clamp(input_energy, min=1e-10))
        
        assert abs(energy_ratio - 1.0) < 1e-3, f"Energy conservation failed for large batch"
    
    def test_zero_input(self, device):
        """Test: Input de ceros."""
        mzi = MZILayer(in_features=4, out_features=4, device=device)
        zero_input = torch.zeros(5, 4, device=device, dtype=torch.float32)
        
        output = mzi(zero_input)
        
        # Output debe ser cero también
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6), "Zero input should give zero output"
    
    def test_dtype_consistency(self, device):
        """Test: Consistencia de dtypes."""
        mzi = MZILayer(in_features=4, out_features=4, device=device, dtype=torch.float32)
        
        # Test diferentes input dtypes
        input_float32 = torch.randn(3, 4, device=device, dtype=torch.float32)
        output_float32 = mzi(input_float32)
        assert output_float32.dtype == torch.float32
        
        # Test conversión automática
        input_float64 = torch.randn(3, 4, device=device, dtype=torch.float64)
        output_float64 = mzi(input_float64)
        assert output_float64.dtype == torch.float32  # Debe convertirse al dtype del layer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])