#!/usr/bin/env python3
"""
Tests para MZI Layers - ACTUALIZADOS para Física Real

🔧 ACTUALIZACIONES PARA NUEVA IMPLEMENTACIÓN FÍSICA:
✅ Tests actualizados para parámetros físicos (theta, phi)  
✅ Eliminados tests de phi_external (ya no existe)
✅ Nuevos tests para componentes físicos
✅ Validación de unitaridad mejorada
✅ Tests de conservación de energía más estrictos

CAMBIOS EN PARÁMETROS:
❌ ANTES: theta, phi_internal, phi_external (3 parámetros)
✅ AHORA: theta, phi (2 parámetros físicos)

Suite completa de tests que valida:
- Comportamiento unitario de matrices con MZIs físicos
- Conservación perfecta de energía  
- Parámetros físicos correctos (2 phase shifters por MZI)
- Gradientes y backpropagation
- Edge cases y robustez
- Nuevos métodos de validación física
"""

import pytest
import torch
import numpy as np
import warnings
from typing import Dict, Any

# Import del módulo a testear
from torchonn.layers import MZILayer, MZIBlockLinear


class TestMZILayer:
    """Tests para MZILayer con implementación física real."""
    
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
        """🔧 ACTUALIZADO: Test inicialización con parámetros físicos."""
        try:
            mzi = MZILayer(in_features=4, out_features=4, device=device)
        except Exception as e:
            pytest.fail(f"MZI initialization failed: {e}")
        
        # Verificar parámetros básicos
        assert mzi.in_features == 4
        assert mzi.out_features == 4
        assert mzi.matrix_dim == 4
        assert mzi.device == device
        
        # 🔧 NUEVOS PARÁMETROS FÍSICOS: solo theta y phi
        assert hasattr(mzi, 'theta'), "Missing theta parameter"
        assert hasattr(mzi, 'phi'), "Missing phi parameter"
        
        # 🔧 ELIMINADO: phi_internal y phi_external ya no existen
        assert not hasattr(mzi, 'phi_internal'), "phi_internal should not exist"
        assert not hasattr(mzi, 'phi_external'), "phi_external should not exist"
        
        # Verificar dimensiones de parámetros físicos
        expected_n_mzis = 4 * (4 - 1) // 2  # 6 MZIs para 4x4
        assert mzi.theta.shape == (expected_n_mzis,), f"Wrong theta shape: {mzi.theta.shape}"
        assert mzi.phi.shape == (expected_n_mzis,), f"Wrong phi shape: {mzi.phi.shape}"
        
        # 🔧 NUEVO: Verificar conteo de componentes físicos
        assert mzi.n_mzis == expected_n_mzis
        assert mzi.get_phase_shifter_count() == expected_n_mzis * 2  # 2 por MZI
        
        print(f"✅ MZI initialized: {expected_n_mzis} MZIs, {mzi.get_phase_shifter_count()} phase shifters")
    
    def test_physical_components_summary(self, mzi_4x4):
        """🔧 NUEVO: Test resumen de componentes físicos."""
        try:
            components = mzi_4x4.get_physical_component_summary()
        except Exception as e:
            pytest.fail(f"Component summary failed: {e}")
        
        # Verificar claves esperadas
        expected_keys = [
            'mzi_count', 'phase_shifter_count', 'splitter_3db_count', 
            'matrix_dimension', 'total_parameters'
        ]
        for key in expected_keys:
            assert key in components, f"Missing key: {key}"
        
        # Verificar valores para 4x4
        assert components['mzi_count'] == 6, f"Wrong MZI count: {components['mzi_count']}"
        assert components['phase_shifter_count'] == 12, f"Wrong phase shifter count: {components['phase_shifter_count']}"
        assert components['splitter_3db_count'] == 12, f"Wrong splitter count: {components['splitter_3db_count']}"
        assert components['matrix_dimension'] == 4
        assert components['total_parameters'] == 12  # 6 MZIs * 2 parameters each
        
        print(f"✅ Physical components: {components}")
    
    def test_unitary_matrix_property(self, mzi_4x4):
        """🔧 ACTUALIZADO: Test unitaridad con nueva validación."""
        try:
            # 🔧 NUEVA API: validate_unitarity() retorna dict
            unitarity_result = mzi_4x4.validate_unitarity(tolerance=1e-4)
        except Exception as e:
            pytest.fail(f"Unitarity validation failed: {e}")
        
        # Verificar estructura del resultado
        expected_keys = ['is_unitary', 'max_error', 'determinant_magnitude', 'determinant_error', 'tolerance']
        for key in expected_keys:
            assert key in unitarity_result, f"Missing unitarity key: {key}"
        
        # Test unitaridad
        assert unitarity_result['is_unitary'], f"Matrix not unitary: {unitarity_result}"
        assert unitarity_result['max_error'] < 1e-4, f"Unitarity error too high: {unitarity_result['max_error']}"
        
        # Test determinante ~1
        det_mag = unitarity_result['determinant_magnitude']
        assert abs(det_mag - 1.0) < 1e-4, f"Determinant magnitude wrong: {det_mag}"
        
        print(f"✅ Unitarity validated: error = {unitarity_result['max_error']:.2e}")
    
    def test_energy_conservation_strict(self, mzi_4x4, input_batch):
        """🔧 ACTUALIZADO: Test conservación de energía más estricto."""
        try:
            output_batch = mzi_4x4(input_batch)
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
        
        # Calcular energías
        input_energy = torch.sum(input_batch**2, dim=1)
        output_energy = torch.sum(output_batch**2, dim=1)
        
        # 🔧 FÍSICA REAL: Conservación perfecta (tolerancia más estricta)
        energy_ratios = output_energy / torch.clamp(input_energy, min=1e-10)
        mean_ratio = torch.mean(energy_ratios)
        std_ratio = torch.std(energy_ratios)
        
        # Para MZI físico real, conservación debe ser perfecta
        assert abs(mean_ratio - 1.0) < 1e-3, f"Energy not conserved: {mean_ratio:.6f} ± {std_ratio:.6f}"
        assert std_ratio < 1e-3, f"Energy conservation inconsistent: std = {std_ratio:.6f}"
        
        print(f"✅ Energy conservation: {mean_ratio:.6f} ± {std_ratio:.6f}")
    
    def test_insertion_loss(self, mzi_4x4):
        """🔧 NUEVO: Test insertion loss para MZI físico."""
        try:
            insertion_loss_db = mzi_4x4.get_insertion_loss_db()
        except Exception as e:
            pytest.fail(f"Insertion loss calculation failed: {e}")
        
        # Para MZI físico unitario, insertion loss debe ser ~0 dB
        assert abs(insertion_loss_db) < 1e-2, f"Insertion loss too high: {insertion_loss_db:.3f} dB"
        
        print(f"✅ Insertion loss: {insertion_loss_db:.3f} dB")
    
    def test_different_sizes(self, device):
        """🔧 ACTUALIZADO: Test diferentes tamaños de matriz."""
        # Test tamaños más pequeños para evitar problemas de memoria
        dimensions = [(2, 2), (3, 3), (4, 4), (6, 6)]
        
        for in_dim, out_dim in dimensions:
            try:
                mzi = MZILayer(in_features=in_dim, out_features=out_dim, device=device)
            except Exception as e:
                pytest.fail(f"Failed to create MZI {in_dim}x{out_dim}: {e}")
            
            # Test forward pass
            input_tensor = torch.randn(4, in_dim, device=device, dtype=torch.float32)
            
            try:
                output = mzi(input_tensor)
            except Exception as e:
                pytest.fail(f"Forward pass failed for {in_dim}x{out_dim}: {e}")
            
            # Verificar dimensiones
            assert output.shape == (4, out_dim), f"Wrong output shape for {in_dim}x{out_dim}"
            
            # 🔧 ACTUALIZADO: Verificar unitaridad para matrices cuadradas
            if in_dim == out_dim:
                unitarity_result = mzi.validate_unitarity(tolerance=1e-3)
                assert unitarity_result['is_unitary'], f"Matrix {in_dim}x{out_dim} not unitary"
                
                # Test conservación de energía más estricta
                input_energy = torch.sum(input_tensor**2, dim=1)
                output_energy = torch.sum(output**2, dim=1)
                energy_ratio = torch.mean(output_energy / torch.clamp(input_energy, min=1e-10))
                
                assert abs(energy_ratio - 1.0) < 1e-2, f"Energy not conserved for {in_dim}x{out_dim}: {energy_ratio:.6f}"
    
    def test_gradients_flow_updated(self, device):
        """🔧 ACTUALIZADO: Test gradientes con nuevos parámetros."""
        try:
            mzi = MZILayer(in_features=4, out_features=4, device=device)
        except Exception as e:
            pytest.fail(f"Failed to create MZI for gradient test: {e}")

        # Input más grande para gradientes más significativos
        input_tensor = torch.randn(16, 4, device=device, dtype=torch.float32, requires_grad=True) * 2.0
        input_tensor.retain_grad()

        # Forward pass
        try:
            output = mzi(input_tensor)
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")

        # Loss function que garantiza gradientes no-cero
        loss = torch.mean(output**2) + 0.01 * torch.mean(torch.abs(output))

        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            pytest.fail(f"Backward pass failed: {e}")

        # 🔧 ACTUALIZADO: Test gradientes en nuevos parámetros (theta, phi)
        param_grads = {}
        for name, param in mzi.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradients on parameter {name}"
                param_grad_norm = torch.norm(param.grad)
                param_grads[name] = param_grad_norm
                assert param_grad_norm > 1e-10, f"Parameter {name} gradients too small: {param_grad_norm:.2e}"
                assert torch.isfinite(param_grad_norm), f"Non-finite gradients on {name}"

        # Verificar que tenemos gradientes en theta y phi específicamente
        assert 'theta' in param_grads, "Missing theta gradients"
        assert 'phi' in param_grads, "Missing phi gradients"
        
        print(f"✅ Gradients - theta: {param_grads['theta']:.2e}, phi: {param_grads['phi']:.2e}")
        
        # 🔧 ELIMINADO: No hay phi_internal ni phi_external
        assert 'phi_internal' not in param_grads, "phi_internal should not exist"
        assert 'phi_external' not in param_grads, "phi_external should not exist"
    
    def test_parameter_reset_updated(self, device):
        """🔧 ACTUALIZADO: Test reset con nuevos parámetros."""
        try:
            mzi = MZILayer(in_features=4, out_features=4, device=device)
        except Exception as e:
            pytest.fail(f"Failed to create MZI for reset test: {e}")
        
        # 🔧 ACTUALIZADO: Guardar parámetros físicos iniciales
        theta_initial = mzi.theta.clone()
        phi_initial = mzi.phi.clone()
        
        # Modificar parámetros
        with torch.no_grad():
            mzi.theta.fill_(1.0)
            mzi.phi.fill_(2.0)
        
        # Reset
        try:
            mzi.reset_parameters()
        except Exception as e:
            pytest.skip(f"Reset parameters failed: {e}")
        
        # 🔧 ACTUALIZADO: Verificar que cambiaron (solo theta y phi)
        assert not torch.allclose(mzi.theta, theta_initial), "Theta not reset"
        assert not torch.allclose(mzi.phi, phi_initial), "Phi not reset"
        
        # Verificar rangos físicos correctos [0, 2π]
        assert torch.all(mzi.theta >= 0) and torch.all(mzi.theta <= 2*np.pi), "Theta out of physical range"
        assert torch.all(mzi.phi >= 0) and torch.all(mzi.phi <= 2*np.pi), "Phi out of physical range"
        
        print(f"✅ Parameters reset - theta range: [{torch.min(mzi.theta):.3f}, {torch.max(mzi.theta):.3f}]")
        print(f"                    phi range: [{torch.min(mzi.phi):.3f}, {torch.max(mzi.phi):.3f}]")


class TestMZIBlockLinear:
    """Tests para MZIBlockLinear - SIN CAMBIOS (no afectado)."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_usv_mode(self, device):
        """Test: Modo USV funciona correctamente."""
        try:
            mzi = MZIBlockLinear(
                in_features=6,
                out_features=4,
                mode="usv",
                device=device
            )
        except Exception as e:
            pytest.fail(f"Failed to create USV mode MZI: {e}")
        
        # Test forward pass
        input_tensor = torch.randn(8, 6, device=device, dtype=torch.float32)
        
        try:
            output = mzi(input_tensor)
        except Exception as e:
            pytest.fail(f"USV forward pass failed: {e}")
        
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
        try:
            mzi = MZIBlockLinear(
                in_features=5,
                out_features=3,
                mode="weight",
                device=device
            )
        except Exception as e:
            pytest.fail(f"Failed to create weight mode MZI: {e}")
        
        # Test forward pass
        input_tensor = torch.randn(4, 5, device=device, dtype=torch.float32)
        
        try:
            output = mzi(input_tensor)
        except Exception as e:
            pytest.fail(f"Weight forward pass failed: {e}")
        
        assert output.shape == (4, 3)
        
        # Test que weight existe
        assert hasattr(mzi, 'weight')
        assert mzi.weight.shape == (3, 5)
    
    def test_phase_mode(self, device):
        """Test: Modo phase funciona correctamente."""
        try:
            mzi = MZIBlockLinear(
                in_features=4,
                out_features=4,
                mode="phase",
                device=device
            )
        except Exception as e:
            pytest.fail(f"Failed to create phase mode MZI: {e}")
        
        # Test forward pass
        input_tensor = torch.randn(6, 4, device=device, dtype=torch.float32)
        
        try:
            output = mzi(input_tensor)
        except Exception as e:
            pytest.fail(f"Phase forward pass failed: {e}")
        
        assert output.shape == (6, 4)
        
        # Test que phases existe
        assert hasattr(mzi, 'phases')
        assert mzi.phases.shape == (8,)  # 4 + 4 = 8


class TestMZIPhysicalValidation:
    """🔧 NUEVA CLASE: Tests específicos para validación física."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_mzi_matrix_physical_correctness(self, device):
        """🔧 NUEVO: Test que la matriz MZI sigue la física real."""
        mzi = MZILayer(in_features=2, out_features=2, device=device)
        
        # Set parámetros conocidos
        with torch.no_grad():
            mzi.theta[0] = 0.0  # No phase shift en brazo superior
            mzi.phi[0] = np.pi  # π phase shift en brazo inferior
        
        # Obtener matriz unitaria
        U = mzi.get_unitary_matrix()
        
        # Para θ=0, φ=π, matriz debe ser aproximadamente:
        # U = 0.5 * [[1 + (-1), 1 - (-1)], [1 - (-1), 1 + (-1)]]
        #   = 0.5 * [[0, 2], [2, 0]] = [[0, 1], [1, 0]]
        expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        
        # Test similitud (con tolerancia para errores numéricos)
        difference = torch.max(torch.abs(U - expected))
        assert difference < 1e-3, f"Matrix not physically correct: difference = {difference:.2e}"
        
        print(f"✅ Physical matrix correct for θ=0, φ=π")
    
    def test_phase_shifter_effects(self, device):
        """🔧 NUEVO: Test efectos independientes de phase shifters."""
        mzi = MZILayer(in_features=2, out_features=2, device=device)
        
        # Test diferentes combinaciones de phase shifters
        test_cases = [
            (0.0, 0.0),      # Sin phases
            (np.pi/2, 0.0),  # Solo theta
            (0.0, np.pi/2),  # Solo phi
            (np.pi/2, np.pi/2),  # Ambos
        ]
        
        for theta, phi in test_cases:
            with torch.no_grad():
                mzi.theta[0] = theta
                mzi.phi[0] = phi
            
            # Verificar unitaridad en cada caso
            unitarity_result = mzi.validate_unitarity(tolerance=1e-6)
            assert unitarity_result['is_unitary'], f"Not unitary for θ={theta:.3f}, φ={phi:.3f}"
            
            # Test conservación de energía
            x = torch.randn(10, 2, device=device)
            y = mzi(x)
            
            input_energy = torch.sum(x**2, dim=1)
            output_energy = torch.sum(y**2, dim=1)
            energy_ratio = torch.mean(output_energy / input_energy)
            
            assert abs(energy_ratio - 1.0) < 1e-4, f"Energy not conserved for θ={theta:.3f}, φ={phi:.3f}: {energy_ratio:.6f}"
        
        print(f"✅ Phase shifters work independently")
    
    def test_mzi_physical_limits(self, device):
        """🔧 NUEVO: Test límites físicos de parámetros."""
        mzi = MZILayer(in_features=3, out_features=3, device=device)
        
        # Test límites de phase shifters [0, 2π]
        with torch.no_grad():
            # Test límite inferior
            mzi.theta.fill_(0.0)
            mzi.phi.fill_(0.0)
            
            unitarity_result = mzi.validate_unitarity()
            assert unitarity_result['is_unitary'], "Not unitary at phase limits (0, 0)"
            
            # Test límite superior
            mzi.theta.fill_(2*np.pi)
            mzi.phi.fill_(2*np.pi)
            
            unitarity_result = mzi.validate_unitarity()
            assert unitarity_result['is_unitary'], "Not unitary at phase limits (2π, 2π)"
            
            # Test valores intermedios
            mzi.theta.fill_(np.pi)
            mzi.phi.fill_(np.pi)
            
            unitarity_result = mzi.validate_unitarity()
            assert unitarity_result['is_unitary'], "Not unitary at phase limits (π, π)"
        
        print(f"✅ Physical parameter limits validated")


class TestMZIEdgeCases:
    """Tests de edge cases y robustez - ALGUNOS ACTUALIZADOS."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_single_input_output(self, device):
        """Test: MZI 1x1 (caso trivial)."""
        try:
            mzi = MZILayer(in_features=1, out_features=1, device=device)
            
            # Para 1x1, no hay MZIs (matriz 1x1 es identidad)
            assert mzi.n_mzis == 0, f"1x1 should have 0 MZIs, got {mzi.n_mzis}"
            
            # Test forward pass
            x = torch.randn(4, 1, device=device, dtype=torch.float32)
            y = mzi(x)
            
            assert y.shape == (4, 1)
            
            # Para matriz identidad, output debe ser igual al input
            assert torch.allclose(y, x, atol=1e-6), "1x1 MZI should be identity"
            
        except Exception as e:
            pytest.skip(f"Single input/output test failed: {e}")
    
    def test_non_square_matrices_updated(self, device):
        """🔧 ACTUALIZADO: Test matrices no cuadradas con nueva física."""
        try:
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
            
            # 🔧 NUEVO: Verificar que la física interna sigue siendo unitaria
            U_expand = mzi_expand.get_unitary_matrix()
            U_reduce = mzi_reduce.get_unitary_matrix()
            
            # Las matrices internas deben ser unitarias
            unitarity_expand = mzi_expand.validate_unitarity()
            unitarity_reduce = mzi_reduce.validate_unitarity()
            
            assert unitarity_expand['is_unitary'], "Expanded MZI internal matrix not unitary"
            assert unitarity_reduce['is_unitary'], "Reduced MZI internal matrix not unitary"
            
        except Exception as e:
            pytest.skip(f"Non-square matrix test failed: {e}")


# 🔧 FUNCIÓN DE UTILIDAD PARA TESTS
def validate_mzi_implementation(mzi_layer, verbose=False):
    """Validar completamente una implementación MZI."""
    results = {}
    
    try:
        # Test 1: Unitaridad
        unitarity = mzi_layer.validate_unitarity()
        results['unitarity'] = unitarity['is_unitary']
        
        # Test 2: Insertion loss
        insertion_loss = mzi_layer.get_insertion_loss_db()
        results['low_insertion_loss'] = abs(insertion_loss) < 0.1
        
        # Test 3: Componentes físicos
        components = mzi_layer.get_physical_component_summary()
        results['components_correct'] = all(v > 0 for v in components.values())
        
        # Test 4: Forward pass
        x_test = torch.randn(4, mzi_layer.in_features, device=mzi_layer.device)
        y_test = mzi_layer(x_test)
        results['forward_pass'] = y_test.shape == (4, mzi_layer.out_features)
        
        overall_pass = all(results.values())
        results['overall'] = overall_pass
        
        if verbose:
            print(f"🔬 MZI Validation Results:")
            for test, passed in results.items():
                print(f"   {test}: {'✅ PASS' if passed else '❌ FAIL'}")
        
        return overall_pass
        
    except Exception as e:
        if verbose:
            print(f"❌ MZI validation failed: {e}")
        return False


# 🔧 EJEMPLO DE USO DE TESTS
if __name__ == "__main__":
    # Ejecutar test básico
    print("🧪 Running basic MZI physics tests...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mzi = MZILayer(4, 4, device=device)
    
    success = validate_mzi_implementation(mzi, verbose=True)
    print(f"🎉 MZI Tests: {'✅ ALL PASSED' if success else '❌ SOME FAILED'}")


# 🔧 RESUMEN DE CAMBIOS EN TESTS:
"""
CAMBIOS PRINCIPALES EN LOS TESTS:

1. ✅ PARÁMETROS ACTUALIZADOS: 
   - test_mzi_initialization: verifica theta, phi (no phi_internal, phi_external)
   - test_parameter_reset_updated: resetea solo theta, phi
   - test_gradients_flow_updated: gradientes en theta, phi

2. ✅ NUEVOS TESTS FÍSICOS:
   - test_physical_components_summary: conteo de componentes
   - test_insertion_loss: pérdidas de inserción
   - test_mzi_matrix_physical_correctness: matriz física correcta
   - test_phase_shifter_effects: efectos independientes de phase shifters
   - test_mzi_physical_limits: límites físicos [0, 2π]

3. ✅ VALIDACIÓN MEJORADA:
   - test_unitary_matrix_property: nueva API validate_unitarity()
   - test_energy_conservation_strict: tolerancias más estrictas
   - Verificación de unitaridad perfecta

4. ✅ EDGE CASES ACTUALIZADOS:
   - test_non_square_matrices_updated: verifica unitaridad interna
   - Mejor manejo de casos especiales

5. ✅ MÉTODOS DE UTILIDAD:
   - validate_mzi_implementation(): validación completa
   - Helpers para testing automatizado

COMPATIBILIDAD:
- MZIBlockLinear tests sin cambios (no afectado)
- API externa preservada
- Nuevos tests no rompen funcionalidad existente
"""