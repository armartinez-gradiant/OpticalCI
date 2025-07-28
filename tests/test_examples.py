"""
Tests para Examples - PtONN-TESTS

Suite de tests que valida:
- Funcionamiento del Example1.py completo
- Todos los demos individuales
- Validación física automática
- Coherencia de resultados
- Performance de ejemplos
"""

import pytest
import torch
import numpy as np
import sys
import os
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Añadir path para importar examples
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.join(os.path.dirname(current_dir), 'examples')
sys.path.insert(0, examples_dir)

# Import del ejemplo
try:
    from Example1 import PhotonicSimulationDemo
except ImportError:
    # Fallback si no se encuentra el archivo
    PhotonicSimulationDemo = None


class TestPhotonicSimulationDemo:
    """Tests para la demo completa de simulación fotónica."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def demo_instance(self, device):
        """Fixture para instancia de demo."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
        return PhotonicSimulationDemo(device=device)
    
    def test_demo_initialization(self, demo_instance, device):
        """Test: Inicialización correcta de la demo."""
        assert demo_instance.device == device
        assert hasattr(demo_instance, 'device')
    
    def test_demo_1_mzi_unitary(self, demo_instance):
        """Test: Demo 1 - MZI unitary behavior funciona."""
        results = demo_instance.demo_1_mzi_unitary_behavior()
        
        # Verificar que retorna resultados
        assert results is not None
        assert isinstance(results, dict)
        
        # Verificar métricas esperadas
        expected_keys = ['energy_conservation', 'energy_std', 'unitarity_error', 'insertion_loss_db']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Validar valores físicos
        energy_conservation = results['energy_conservation']
        assert abs(energy_conservation - 1.0) < 0.01, f"Energy not conserved: {energy_conservation:.6f}"
        
        unitarity_error = results['unitarity_error']
        assert unitarity_error < 1e-4, f"Unitarity error too high: {unitarity_error:.2e}"
        
        insertion_loss = results['insertion_loss_db']
        assert abs(insertion_loss) < 2.0, f"Insertion loss too high: {insertion_loss:.3f} dB"
    
    def test_demo_2_microring_spectral(self, demo_instance):
        """Test: Demo 2 - Microring spectral response funciona."""
        results = demo_instance.demo_2_microring_spectral_response()
        
        # Verificar que retorna resultados
        assert results is not None
        assert isinstance(results, dict)
        
        # Verificar métricas esperadas
        expected_keys = [
            'resonance_wavelength_nm', 'extinction_ratio_db', 'extinction_ratio_theory_db',
            'fsr_theoretical_pm', 'energy_conservation', 'validation'
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Validar valores físicos
        resonance_wl = results['resonance_wavelength_nm']
        assert 1549 < resonance_wl < 1551, f"Resonance not at 1550nm: {resonance_wl:.3f} nm"
        
        extinction_ratio = results['extinction_ratio_db']
        assert 5 < extinction_ratio < 35, f"ER not realistic: {extinction_ratio:.1f} dB"
        
        energy_conservation = results['energy_conservation']
        assert 0.8 < energy_conservation < 1.1, f"Energy not conserved: {energy_conservation:.3f}"
        
        # Validar arrays de respuesta
        assert 'wavelengths_nm' in results
        assert 'through_response' in results
        assert 'drop_response' in results
        
        wavelengths_nm = results['wavelengths_nm']
        through_response = results['through_response']
        drop_response = results['drop_response']
        
        assert len(wavelengths_nm) == len(through_response)
        assert len(wavelengths_nm) == len(drop_response)
        assert np.all(through_response <= 1.01), "Through response > 1.0"
        assert np.all(drop_response <= 1.01), "Drop response > 1.0"
    
    def test_demo_3_add_drop_mrr(self, demo_instance):
        """Test: Demo 3 - Add-Drop MRR funciona."""
        results = demo_instance.demo_3_add_drop_mrr_transfer()
        
        # Verificar que retorna resultados
        assert results is not None
        assert isinstance(results, dict)
        
        # Verificar métricas esperadas
        expected_keys = [
            'wavelengths_nm', 'through_power', 'drop_power',
            'coupling_1', 'coupling_2', 'fsr_pm'
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Validar coupling coefficients
        coupling_1 = results['coupling_1']
        coupling_2 = results['coupling_2']
        
        assert 0.01 < coupling_1 < 0.2, f"Coupling 1 not realistic: {coupling_1:.4f}"
        assert 0.01 < coupling_2 < 0.2, f"Coupling 2 not realistic: {coupling_2:.4f}"
        
        # Validar FSR
        fsr_pm = results['fsr_pm']
        assert 5000 < fsr_pm < 15000, f"FSR not realistic: {fsr_pm:.0f} pm"
        
        # Validar power arrays
        through_power = results['through_power']
        drop_power = results['drop_power']
        
        assert np.all(through_power <= 1.01), "Through power > 1.0"
        assert np.all(drop_power <= 1.01), "Drop power > 1.0"
        assert np.all(through_power >= 0), "Through power < 0"
        assert np.all(drop_power >= 0), "Drop power < 0"
    
    def test_demo_4_wdm_system(self, demo_instance):
        """Test: Demo 4 - WDM system funciona."""
        results = demo_instance.demo_4_wdm_system()
        
        # Verificar que retorna resultados
        assert results is not None
        assert isinstance(results, dict)
        
        # Verificar métricas esperadas
        expected_keys = ['n_channels', 'wavelengths_nm', 'fidelities', 'avg_fidelity']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Validar número de canales
        n_channels = results['n_channels']
        assert n_channels == 4, f"Wrong number of channels: {n_channels}"
        
        # Validar wavelengths
        wavelengths_nm = results['wavelengths_nm']
        assert len(wavelengths_nm) == 4
        expected_wls = [1530, 1540, 1550, 1560]
        for actual, expected in zip(wavelengths_nm, expected_wls):
            assert abs(actual - expected) < 1, f"Wavelength mismatch: {actual} vs {expected}"
        
        # Validar fidelidades
        fidelities = results['fidelities']
        avg_fidelity = results['avg_fidelity']
        
        assert len(fidelities) == 4
        assert all(f > 0.7 for f in fidelities), f"Low fidelities: {fidelities}"
        assert avg_fidelity > 0.8, f"Low average fidelity: {avg_fidelity:.3f}"
    
    def test_demo_5_complete_network(self, demo_instance):
        """Test: Demo 5 - Complete photonic network funciona."""
        results = demo_instance.demo_5_complete_photonic_network()
        
        # Verificar que retorna resultados
        assert results is not None
        assert isinstance(results, dict)
        
        # Verificar métricas esperadas
        expected_keys = ['input_shape', 'output_shape', 'forward_time_ms', 'output_stats']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Validar shapes
        input_shape = results['input_shape']
        output_shape = results['output_shape']
        
        assert input_shape == [32, 8], f"Wrong input shape: {input_shape}"
        assert output_shape[0] == 32, f"Wrong batch size: {output_shape}"
        assert len(output_shape) == 2, f"Wrong output dimensions: {output_shape}"
        
        # Validar timing
        forward_time_ms = results['forward_time_ms']
        assert forward_time_ms < 5000, f"Network too slow: {forward_time_ms:.1f} ms"  # < 5 seconds
        
        # Validar estadísticas de output
        output_stats = results['output_stats']
        required_stats = ['mean', 'std', 'range']
        for stat in required_stats:
            assert stat in output_stats, f"Missing output stat: {stat}"
            assert not np.isnan(output_stats[stat]), f"NaN in output stat: {stat}"
            assert not np.isinf(output_stats[stat]), f"Inf in output stat: {stat}"


class TestDemoIntegration:
    """Tests de integración entre demos."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def demo_instance(self, device):
        """Fixture para instancia de demo."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
        return PhotonicSimulationDemo(device=device)
    
    def test_run_all_demos(self, demo_instance):
        """Test: Ejecutar todas las demos juntas."""
        # Este test puede ser lento
        results = demo_instance.run_all_demos()
        
        # Verificar que todas las demos corrieron
        assert results is not None
        assert isinstance(results, dict)
        
        expected_demos = ['mzi', 'microring', 'add_drop', 'wdm', 'complete_network']
        for demo in expected_demos:
            assert demo in results, f"Demo {demo} not found in results"
            assert results[demo] is not None, f"Demo {demo} returned None"
    
    def test_physics_validation_coherence(self, demo_instance):
        """Test: Validación física coherente entre demos."""
        # Ejecutar solo los demos con validación física
        mzi_results = demo_instance.demo_1_mzi_unitary_behavior()
        microring_results = demo_instance.demo_2_microring_spectral_response()
        
        # Verificar coherencia en conservación de energía
        mzi_energy = mzi_results['energy_conservation']
        microring_validation = microring_results['validation']
        microring_energy = microring_validation['energy_conserved']
        
        # MZI debe conservar energía perfectamente (unitario)
        assert abs(mzi_energy - 1.0) < 0.01, "MZI energy not conserved"
        
        # Microring debe pasar validación física
        assert microring_energy, "Microring energy not conserved"
    
    def test_parameter_consistency(self, demo_instance):
        """Test: Consistencia de parámetros entre demos."""
        # Ejecutar demos que usan microrings
        microring_results = demo_instance.demo_2_microring_spectral_response()
        add_drop_results = demo_instance.demo_3_add_drop_mrr_transfer()
        
        # Verificar FSR scaling con radio
        # Microring: R=10μm, Add-Drop: R=8μm
        microring_fsr = microring_results['fsr_theoretical_pm']
        add_drop_fsr = add_drop_results['fsr_pm']
        
        # FSR ∝ 1/R, por lo que FSR(8μm) > FSR(10μm)
        assert add_drop_fsr > microring_fsr, (
            f"FSR scaling incorrect: {add_drop_fsr:.0f} pm (8μm) should be > "
            f"{microring_fsr:.0f} pm (10μm)"
        )


class TestDemoPerformance:
    """Tests de performance de las demos."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def demo_instance(self, device):
        """Fixture para instancia de demo."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
        return PhotonicSimulationDemo(device=device)
    
    @pytest.mark.slow
    def test_demo_execution_times(self, demo_instance):
        """Test: Tiempos de ejecución de demos son razonables."""
        import time
        
        demo_methods = [
            ('MZI', demo_instance.demo_1_mzi_unitary_behavior),
            ('Microring', demo_instance.demo_2_microring_spectral_response),
            ('Add-Drop', demo_instance.demo_3_add_drop_mrr_transfer),
            ('WDM', demo_instance.demo_4_wdm_system),
            ('Complete Network', demo_instance.demo_5_complete_photonic_network)
        ]
        
        execution_times = {}
        
        for name, method in demo_methods:
            start_time = time.time()
            result = method()
            execution_time = time.time() - start_time
            
            execution_times[name] = execution_time
            
            # Verificar que el demo funcionó
            assert result is not None, f"Demo {name} failed"
            
            # Verificar tiempo razonable (< 30 segundos por demo)
            assert execution_time < 30.0, f"Demo {name} too slow: {execution_time:.2f}s"
        
        # Verificar tiempo total razonable
        total_time = sum(execution_times.values())
        assert total_time < 120.0, f"Total demo time too long: {total_time:.2f}s"
        
        print(f"\n📊 Demo Execution Times:")
        for name, time_s in execution_times.items():
            print(f"   {name}: {time_s:.2f}s")
        print(f"   Total: {total_time:.2f}s")
    
    @pytest.mark.slow
    def test_memory_usage(self, demo_instance, device):
        """Test: Uso de memoria razonable."""
        if device.type != 'cuda':
            pytest.skip("Memory test requires CUDA")
        
        import gc
        
        # Limpiar memoria inicial
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated()
        
        # Ejecutar demo más pesado
        results = demo_instance.demo_2_microring_spectral_response()
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # Verificar uso de memoria razonable (< 500MB)
        assert memory_used < 500e6, f"Excessive memory usage: {memory_used/1e6:.1f} MB"
        
        # Limpiar
        del results
        torch.cuda.empty_cache()
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated()
        memory_leak = final_memory - initial_memory
        
        # Verificar no hay memory leaks significativos (< 50MB)
        assert memory_leak < 50e6, f"Memory leak detected: {memory_leak/1e6:.1f} MB"


class TestDemoErrorHandling:
    """Tests de manejo de errores en demos."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def demo_instance(self, device):
        """Fixture para instancia de demo."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
        return PhotonicSimulationDemo(device=device)
    
    def test_graceful_failure_handling(self, demo_instance):
        """Test: Manejo elegante de fallos."""
        # Simular condición de error parcial
        original_run = demo_instance.run_all_demos
        
        def mock_run_with_failure():
            # Ejecutar algunos demos exitosamente
            results = {}
            try:
                results['mzi'] = demo_instance.demo_1_mzi_unitary_behavior()
                results['microring'] = demo_instance.demo_2_microring_spectral_response() 
                # Simular fallo en demo posterior
                results['add_drop'] = None  # Fallo simulado
                results['wdm'] = demo_instance.demo_4_wdm_system()
                results['complete_network'] = demo_instance.demo_5_complete_photonic_network()
            except Exception as e:
                print(f"Demo failed: {e}")
                return results
            
            return results
        
        # El sistema debe manejar fallos parciales elegantemente
        results = mock_run_with_failure()
        
        # Verificar que algunos resultados están presentes
        assert results is not None
        assert len(results) > 0
        
        # Verificar que los demos exitosos tienen resultados válidos
        if 'mzi' in results and results['mzi'] is not None:
            assert 'energy_conservation' in results['mzi']
    
    def test_device_consistency(self, device):
        """Test: Consistencia de device en toda la demo."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
            
        demo = PhotonicSimulationDemo(device=device)
        
        # Verificar que todos los componentes están en el device correcto
        # (Esto requeriría acceso a componentes internos)
        assert demo.device == device


class TestExampleOutputValidation:
    """Tests para validar outputs específicos del ejemplo."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_microring_response_shape(self, device):
        """Test: Shape de respuesta espectral del microring."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
            
        demo = PhotonicSimulationDemo(device=device)
        results = demo.demo_2_microring_spectral_response()
        
        # Verificar que las arrays tienen shapes consistentes
        wavelengths = results['wavelengths_nm']
        through = results['through_response']
        drop = results['drop_response']
        
        assert len(wavelengths) == len(through), "Wavelength/through length mismatch"
        assert len(wavelengths) == len(drop), "Wavelength/drop length mismatch"
        assert len(wavelengths) > 100, "Too few wavelength points"
    
    def test_validation_metrics_present(self, device):
        """Test: Métricas de validación están presentes."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
            
        demo = PhotonicSimulationDemo(device=device)
        results = demo.demo_2_microring_spectral_response()
        
        validation = results['validation']
        
        required_metrics = [
            'energy_conserved', 'energy_conservation', 'expected_conservation',
            'extinction_ratio_measured_db', 'extinction_ratio_theory_db',
            'extinction_ratio_coherent', 'resonance_centered', 
            'resonance_wavelength_nm'
        ]
        
        for metric in required_metrics:
            assert metric in validation, f"Missing validation metric: {metric}"
    
    def test_physical_values_realistic(self, device):
        """Test: Valores físicos son realistas."""
        if PhotonicSimulationDemo is None:
            pytest.skip("Example1.py not found")
            
        demo = PhotonicSimulationDemo(device=device)
        results = demo.demo_2_microring_spectral_response()
        
        # Test extinction ratio
        er_measured = results['extinction_ratio_db']
        er_theory = results['extinction_ratio_theory_db']
        
        assert 5 <= er_measured <= 35, f"ER measured not realistic: {er_measured:.1f} dB"
        assert 5 <= er_theory <= 35, f"ER theory not realistic: {er_theory:.1f} dB"
        
        # Test FSR
        fsr_pm = results['fsr_theoretical_pm']
        assert 5000 <= fsr_pm <= 15000, f"FSR not realistic: {fsr_pm:.0f} pm"
        
        # Test resonance wavelength
        resonance_nm = results['validation']['resonance_wavelength_nm']
        assert 1549 <= resonance_nm <= 1551, f"Resonance not near 1550nm: {resonance_nm:.3f} nm"


# Mock para matplotlib si no está disponible
@pytest.fixture(autouse=True)
def mock_matplotlib():
    """Mock matplotlib para evitar dependencias gráficas en tests."""
    with patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.plot'), \
         patch('matplotlib.pyplot.subplot'), \
         patch('matplotlib.pyplot.xlabel'), \
         patch('matplotlib.pyplot.ylabel'), \
         patch('matplotlib.pyplot.title'), \
         patch('matplotlib.pyplot.legend'), \
         patch('matplotlib.pyplot.grid'), \
         patch('matplotlib.pyplot.ylim'), \
         patch('matplotlib.pyplot.axvline'), \
         patch('matplotlib.pyplot.tight_layout'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.show'), \
         patch('matplotlib.pyplot.close'):
        yield


if __name__ == "__main__":
    pytest.main([__file__, "-v"])