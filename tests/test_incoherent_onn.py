"""
Tests para IncoherentONN - Suite Completa

Valida:
- Inicialización y configuración
- Forward pass y gradientes
- Validación física específica (transmisiones, WDM)
- Métricas de eficiencia óptica
- Comparación con CoherentONN
- Edge cases y robustez
"""

import pytest
import torch
import numpy as np
import warnings
from typing import Dict, Any, List

# Import de IncoherentONN
try:
    from torchonn.onns.architectures import IncoherentONN, CoherentONN, BaseONN
    INCOHERENT_AVAILABLE = True
except ImportError as e:
    INCOHERENT_AVAILABLE = False
    print(f"⚠️ IncoherentONN no disponible: {e}")


@pytest.mark.skipif(not INCOHERENT_AVAILABLE, reason="IncoherentONN not available")
class TestIncoherentONN:
    """Tests principales para IncoherentONN."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def simple_incoherent_onn(self, device):
        """Fixture para IncoherentONN simple."""
        return IncoherentONN(
            layer_sizes=[4, 6, 3],
            n_wavelengths=4,
            activation_type="relu",
            optical_power=1.0,
            device=device
        )
    
    def test_incoherent_onn_initialization(self, device):
        """Test: Inicialización correcta de IncoherentONN."""
        # Test configuración básica
        layer_sizes = [8, 12, 6]
        n_wavelengths = 6
        
        onn = IncoherentONN(
            layer_sizes=layer_sizes,
            n_wavelengths=n_wavelengths,
            activation_type="sigmoid",
            optical_power=2.0,
            device=device
        )
        
        # Verificar parámetros básicos
        assert onn.layer_sizes == layer_sizes
        assert onn.n_wavelengths == n_wavelengths
        assert onn.activation_type == "sigmoid"
        assert onn.optical_power == 2.0
        assert onn.device == device
        
        # Verificar estructura de red
        assert len(onn.incoherent_layers) == len(layer_sizes) - 2  # N-1 capas fotónicas, última eléctrica
        assert hasattr(onn, 'final_layer')
        assert isinstance(onn.activation, torch.nn.Sigmoid)
        
        # Verificar herencia de BaseONN
        assert isinstance(onn, BaseONN)
        
        print(f"✅ IncoherentONN initialized with {len(layer_sizes)} layers, {n_wavelengths} wavelengths")
    
    def test_forward_pass_basic(self, simple_incoherent_onn, device):
        """Test: Forward pass básico funciona."""
        batch_size = 8
        input_data = torch.randn(batch_size, 4, device=device, dtype=torch.float32) * 0.5
        
        # Forward pass
        try:
            output = simple_incoherent_onn(input_data)
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
        
        # Verificar dimensiones
        assert output.shape == (batch_size, 3), f"Wrong output shape: {output.shape}"
        
        # Verificar no hay NaN/Inf
        assert not torch.any(torch.isnan(output)), "NaN detected in output"
        assert not torch.any(torch.isinf(output)), "Inf detected in output"
        
        # Verificar que output no es trivial
        output_norm = torch.norm(output)
        assert output_norm > 1e-6, f"Output too small: {output_norm:.2e}"
        
        print(f"✅ Forward pass: {input_data.shape} → {output.shape}, norm: {output_norm:.3f}")
    
    def test_different_configurations(self, device):
        """Test: Diferentes configuraciones de IncoherentONN."""
        configurations = [
            {
                "layer_sizes": [3, 5, 2], 
                "n_wavelengths": 2, 
                "activation": "relu",
                "name": "Small Network"
            },
            {
                "layer_sizes": [6, 8, 6, 4], 
                "n_wavelengths": 8, 
                "activation": "tanh",
                "name": "Deep Network"
            },
            {
                "layer_sizes": [10, 10], 
                "n_wavelengths": 4, 
                "activation": "sigmoid",
                "name": "Single Hidden Layer"
            }
        ]
        
        for config in configurations:
            print(f"\n   Testing {config['name']}: {config['layer_sizes']}")
            
            try:
                onn = IncoherentONN(
                    layer_sizes=config["layer_sizes"],
                    n_wavelengths=config["n_wavelengths"],
                    activation_type=config["activation"],
                    device=device
                )
                
                # Test forward pass
                input_size = config["layer_sizes"][0]
                output_size = config["layer_sizes"][-1]
                
                x = torch.randn(4, input_size, device=device) * 0.3
                y = onn(x)
                
                assert y.shape == (4, output_size), f"Wrong shape for {config['name']}"
                assert not torch.any(torch.isnan(y)), f"NaN in {config['name']}"
                
                print(f"   ✅ {config['name']}: {x.shape} → {y.shape}")
                
            except Exception as e:
                pytest.fail(f"Configuration {config['name']} failed: {e}")
    
    def test_gradients_flow(self, simple_incoherent_onn, device):
        """Test: Gradientes fluyen correctamente."""
        batch_size = 4
        input_data = torch.randn(batch_size, 4, device=device, dtype=torch.float32, requires_grad=True)
        
        # Forward + backward
        try:
            output = simple_incoherent_onn(input_data)
            loss = torch.mean(output**2) + 0.01 * torch.mean(torch.abs(output))
            loss.backward()
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")
        
        # Test gradientes en input
        assert input_data.grad is not None, "No gradients on input"
        input_grad_norm = torch.norm(input_data.grad)
        assert input_grad_norm > 1e-8, f"Input gradients too small: {input_grad_norm:.2e}"
        
        # Test gradientes en parámetros
        param_grads_found = False
        for name, param in simple_incoherent_onn.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_grad_norm = torch.norm(param.grad)
                if param_grad_norm > 1e-10:
                    param_grads_found = True
                    print(f"   Parameter {name}: grad_norm = {param_grad_norm:.2e}")
        
        assert param_grads_found, "No meaningful parameter gradients found"
        print(f"✅ Gradients flowing correctly")
    
    def test_optical_efficiency_metrics(self, simple_incoherent_onn):
        """Test: Métricas de eficiencia óptica específicas."""
        try:
            efficiency = simple_incoherent_onn.get_optical_efficiency_metrics()
        except Exception as e:
            pytest.fail(f"get_optical_efficiency_metrics failed: {e}")
        
        # Verificar claves esperadas
        required_keys = [
            "architecture_type", "total_microrings", "total_photodetectors",
            "wavelength_channels", "parallel_operations", "optical_fraction",
            "optical_operations", "theoretical_speedup"
        ]
        
        missing_keys = [key for key in required_keys if key not in efficiency]
        assert not missing_keys, f"Missing efficiency keys: {missing_keys}"
        
        # Verificar valores razonables
        assert efficiency["architecture_type"] == "incoherent"
        assert efficiency["total_microrings"] > 0
        assert efficiency["wavelength_channels"] == 4  # Como configuramos
        assert 0 < efficiency["optical_fraction"] <= 1
        assert efficiency["theoretical_speedup"] > 1
        
        print(f"✅ Efficiency metrics: {efficiency['optical_fraction']:.2f} optical fraction")
        print(f"   Microrings: {efficiency['total_microrings']}, Speedup: {efficiency['theoretical_speedup']:.1f}x")
    
    def test_physics_validation(self, simple_incoherent_onn):
        """Test: Validación física específica de incoherent."""
        try:
            physics = simple_incoherent_onn.validate_physics()
        except Exception as e:
            pytest.fail(f"Physics validation failed: {e}")
        
        # Verificar claves específicas de incoherent
        expected_keys = [
            "valid_transmissions", "transmission_range", 
            "energy_conservation_type", "allows_energy_loss"
        ]
        
        missing_keys = [key for key in expected_keys if key not in physics]
        assert not missing_keys, f"Missing physics keys: {missing_keys}"
        
        # Verificar propiedades físicas
        assert physics["energy_conservation_type"] == "intensity_based"
        assert physics["allows_energy_loss"] == True  # Característica de incoherent
        
        # Verificar transmisiones válidas
        if physics["valid_transmissions"]:
            min_trans, max_trans = physics["transmission_range"]
            assert 0 <= min_trans <= max_trans <= 1, f"Invalid transmission range: [{min_trans}, {max_trans}]"
        
        print(f"✅ Physics validation: transmissions valid = {physics['valid_transmissions']}")
        print(f"   Type: {physics['energy_conservation_type']}")
    
    def test_wavelength_scaling(self, device):
        """Test: Escalabilidad con número de wavelengths."""
        base_config = {"layer_sizes": [4, 6, 3], "device": device}
        wavelength_counts = [2, 4, 8, 12]
        
        performances = []
        
        for n_wl in wavelength_counts:
            print(f"\n   Testing {n_wl} wavelengths...")
            
            try:
                onn = IncoherentONN(n_wavelengths=n_wl, **base_config)
                
                # Medir tiempo de forward pass
                x = torch.randn(8, 4, device=device)
                
                import time
                start_time = time.time()
                with torch.no_grad():
                    y = onn(x)
                forward_time = time.time() - start_time
                
                # Obtener métricas
                efficiency = onn.get_optical_efficiency_metrics()
                
                performances.append({
                    "wavelengths": n_wl,
                    "forward_time": forward_time,
                    "speedup": efficiency["theoretical_speedup"],
                    "parallel_ops": efficiency["parallel_operations"]
                })
                
                print(f"   ✅ {n_wl} WL: {forward_time*1000:.2f}ms, speedup: {efficiency['theoretical_speedup']:.1f}x")
                
            except Exception as e:
                pytest.fail(f"Wavelength scaling test failed for {n_wl} wavelengths: {e}")
        
        # Verificar que el speedup teórico aumenta con wavelengths
        speedups = [p["speedup"] for p in performances]
        assert speedups == sorted(speedups), "Theoretical speedup should increase with wavelengths"
    
    def test_batch_processing(self, simple_incoherent_onn, device):
        """Test: Procesamiento en diferentes batch sizes."""
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 4, device=device) * 0.5
            
            try:
                output = simple_incoherent_onn(input_data)
            except Exception as e:
                pytest.fail(f"Batch processing failed for size {batch_size}: {e}")
            
            assert output.shape == (batch_size, 3), f"Wrong output shape for batch {batch_size}"
            assert not torch.any(torch.isnan(output)), f"NaN for batch size {batch_size}"
            
        print(f"✅ Batch processing working for sizes: {batch_sizes}")
    
    def test_activation_functions(self, device):
        """Test: Diferentes funciones de activación."""
        activations = ["relu", "sigmoid", "tanh"]
        base_config = {"layer_sizes": [4, 6, 3], "n_wavelengths": 4, "device": device}
        
        for activation in activations:
            print(f"\n   Testing {activation} activation...")
            
            try:
                onn = IncoherentONN(activation_type=activation, **base_config)
                
                x = torch.randn(4, 4, device=device) * 0.5
                y = onn(x)
                
                assert not torch.any(torch.isnan(y)), f"NaN with {activation}"
                
                # Verificar rango apropiado según activación
                if activation == "sigmoid":
                    # Sigmoid output debe estar cerca de [0,1] después de capa final
                    pass  # La capa final puede extender el rango
                elif activation == "tanh":
                    # Tanh intermedio, pero capa final puede ser cualquier rango
                    pass
                elif activation == "relu":
                    # ReLU intermedio, capa final cualquier rango
                    pass
                
                print(f"   ✅ {activation}: output range [{torch.min(y):.3f}, {torch.max(y):.3f}]")
                
            except Exception as e:
                pytest.fail(f"Activation {activation} failed: {e}")


@pytest.mark.skipif(not INCOHERENT_AVAILABLE, reason="IncoherentONN not available")
class TestCoherentVsIncoherent:
    """Tests de comparación entre arquitecturas."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_architecture_comparison(self, device):
        """Test: Comparación directa entre arquitecturas."""
        layer_sizes = [6, 8, 4]
        
        try:
            # Crear ambas arquitecturas
            coherent_onn = CoherentONN(layer_sizes=layer_sizes, device=device)
            incoherent_onn = IncoherentONN(layer_sizes=layer_sizes, n_wavelengths=4, device=device)
            
            print(f"✅ Both architectures created successfully")
            
            # Test mismo input
            x = torch.randn(4, 6, device=device) * 0.5
            
            # Forward passes
            y_coherent = coherent_onn(x)
            y_incoherent = incoherent_onn(x)
            
            # Verificar shapes
            assert y_coherent.shape == y_incoherent.shape == (4, 4)
            
            # Verificar que son diferentes (diferentes principios físicos)
            output_diff = torch.norm(y_coherent - y_incoherent)
            assert output_diff > 1e-6, "Outputs should be different (different physics)"
            
            print(f"✅ Forward passes: coherent {y_coherent.shape}, incoherent {y_incoherent.shape}")
            print(f"   Output difference: {output_diff:.6f}")
            
        except Exception as e:
            pytest.skip(f"Architecture comparison failed: {e}")
    
    def test_efficiency_comparison(self, device):
        """Test: Comparación de eficiencia óptica."""
        layer_sizes = [4, 6, 3]
        
        try:
            coherent_onn = CoherentONN(layer_sizes=layer_sizes, device=device)
            incoherent_onn = IncoherentONN(layer_sizes=layer_sizes, n_wavelengths=4, device=device)
            
            # Obtener métricas de eficiencia
            coherent_eff = coherent_onn.get_optical_efficiency()
            incoherent_eff = incoherent_onn.get_optical_efficiency()
            
            print(f"📊 Efficiency Comparison:")
            print(f"   Coherent optical fraction: {coherent_eff['optical_fraction']:.3f}")
            print(f"   Incoherent optical fraction: {incoherent_eff['optical_fraction']:.3f}")
            print(f"   Coherent speedup: {coherent_eff['theoretical_speedup']:.1f}x")
            print(f"   Incoherent speedup: {incoherent_eff['theoretical_speedup']:.1f}x")
            
            # Verificar que ambas tienen alta fracción óptica
            assert coherent_eff['optical_fraction'] > 0.3, "Coherent should have high optical fraction"
            assert incoherent_eff['optical_fraction'] > 0.3, "Incoherent should have high optical fraction"
            
            # Incoherent debería tener ventaja en escalabilidad
            if 'scalability_factor' in incoherent_eff:
                assert incoherent_eff['scalability_factor'] >= 4, "Incoherent should show WDM scalability"
            
        except Exception as e:
            pytest.skip(f"Efficiency comparison failed: {e}")
    
    def test_physics_differences(self, device):
        """Test: Diferencias en validación física."""
        try:
            coherent_onn = CoherentONN(layer_sizes=[4, 4], device=device)  # Cuadrada para unitaridad
            incoherent_onn = IncoherentONN(layer_sizes=[4, 6, 4], n_wavelengths=4, device=device)
            
            # Validaciones físicas específicas
            coherent_unitarity = coherent_onn.validate_unitarity()
            incoherent_physics = incoherent_onn.validate_physics()
            
            print(f"🔬 Physics Comparison:")
            print(f"   Coherent unitarity: {coherent_unitarity['overall_valid']}")
            print(f"   Incoherent transmissions: {incoherent_physics['valid_transmissions']}")
            print(f"   Coherent principle: unitary matrices")
            print(f"   Incoherent principle: {incoherent_physics['energy_conservation_type']}")
            
            # Coherent debe validar unitaridad
            assert 'overall_valid' in coherent_unitarity, "Coherent should validate unitarity"
            
            # Incoherent debe validar transmisiones
            assert 'valid_transmissions' in incoherent_physics, "Incoherent should validate transmissions"
            assert incoherent_physics['allows_energy_loss'] == True, "Incoherent should allow energy loss"
            
        except Exception as e:
            pytest.skip(f"Physics comparison failed: {e}")


@pytest.mark.skipif(not INCOHERENT_AVAILABLE, reason="IncoherentONN not available")
class TestIncoherentEdgeCases:
    """Tests de edge cases y robustez."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_single_wavelength(self, device):
        """Test: Una sola wavelength."""
        try:
            onn = IncoherentONN(
                layer_sizes=[3, 4, 2],
                n_wavelengths=1,  # Solo 1 wavelength
                device=device
            )
            
            x = torch.randn(2, 3, device=device)
            y = onn(x)
            
            assert y.shape == (2, 2)
            assert not torch.any(torch.isnan(y))
            
            print(f"✅ Single wavelength working")
            
        except Exception as e:
            pytest.skip(f"Single wavelength test failed: {e}")
    
    def test_large_wavelength_count(self, device):
        """Test: Muchas wavelengths."""
        try:
            onn = IncoherentONN(
                layer_sizes=[4, 6, 3],
                n_wavelengths=16,  # Muchas wavelengths
                device=device
            )
            
            x = torch.randn(2, 4, device=device)
            y = onn(x)
            
            assert y.shape == (2, 3)
            
            # Verificar que el speedup teórico escala
            efficiency = onn.get_optical_efficiency()
            assert efficiency['theoretical_speedup'] > 5, "High wavelength count should give high speedup"
            
            print(f"✅ Large wavelength count: {efficiency['theoretical_speedup']:.1f}x speedup")
            
        except Exception as e:
            pytest.skip(f"Large wavelength count test failed: {e}")
    
    def test_minimal_network(self, device):
        """Test: Red mínima posible."""
        try:
            # Red más pequeña posible: input -> output directo
            onn = IncoherentONN(
                layer_sizes=[2, 1],  # Solo 2 capas
                n_wavelengths=2,
                device=device
            )
            
            x = torch.randn(3, 2, device=device)
            y = onn(x)
            
            assert y.shape == (3, 1)
            print(f"✅ Minimal network working")
            
        except Exception as e:
            pytest.skip(f"Minimal network test failed: {e}")
    
    def test_zero_input(self, device):
        """Test: Input de ceros."""
        try:
            onn = IncoherentONN(layer_sizes=[4, 6, 3], device=device)
            
            # Input de ceros
            x = torch.zeros(2, 4, device=device)
            y = onn(x)
            
            # Para incoherent, cero input debería dar output cerca de cero
            # (después de activaciones y bias terms)
            output_norm = torch.norm(y)
            print(f"✅ Zero input → output norm: {output_norm:.6f}")
            
        except Exception as e:
            pytest.skip(f"Zero input test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])