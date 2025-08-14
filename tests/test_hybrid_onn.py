#!/usr/bin/env python3
"""
Tests Completos para HybridONN

Suite de tests basada en los resultados exitosos de la demo.
UBICACIÓN: tests/test_hybrid_onn.py
"""

import pytest
import torch
import numpy as np
import time
import warnings

# Configurar warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Test if HybridONN is available
HYBRID_AVAILABLE = False
try:
    from torchonn.onns.architectures import HybridONN, HybridMode
    HYBRID_AVAILABLE = True
except ImportError:
    HybridONN = None
    HybridMode = None


class TestHybridONN:
    """Tests completos para HybridONN basados en demo exitosa."""
    
    @pytest.fixture
    def device(self):
        """Device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture  
    def layer_sizes(self):
        """Layer sizes de la demo exitosa."""
        return [8, 12, 8, 4]
    
    @pytest.mark.skipif(not HYBRID_AVAILABLE, reason="HybridONN not available")
    def test_hybrid_modes_from_demo(self, device, layer_sizes):
        """Test todos los modos híbridos validados en la demo."""
        
        modes_to_test = [
            HybridMode.PURE_COHERENT,
            HybridMode.PURE_INCOHERENT,
            HybridMode.ALTERNATING,
            HybridMode.FRONT_COHERENT,
            HybridMode.ADAPTIVE
        ]
        
        for mode in modes_to_test:
            onn = HybridONN(
                layer_sizes=layer_sizes,
                hybrid_mode=mode,
                device=device,
                n_wavelengths=4
            )
            
            # Forward pass básico
            x = torch.randn(4, layer_sizes[0], device=device) * 0.5
            y = onn(x)
            
            # Validaciones básicas
            assert y.shape == (4, layer_sizes[-1])
            assert not torch.any(torch.isnan(y))
            assert not torch.any(torch.isinf(y))
            
            # Validar métricas de la demo
            metrics = onn.get_hybrid_metrics()
            assert metrics["architecture_type"] == "HybridONN"
            assert metrics["hybrid_mode"] == mode.value
            
            # Scores de rendimiento (de la demo)
            perf = metrics["performance_estimates"]
            assert 1.0 <= perf["precision_score"] <= 10.0
            assert 1.0 <= perf["scalability_score"] <= 10.0
            assert 10.0 <= perf["balanced_score"] <= 20.0
    
    @pytest.mark.skipif(not HYBRID_AVAILABLE, reason="HybridONN not available")
    def test_transition_physics_from_demo(self, device):
        """Test física de transiciones validada en la demo."""
        
        layer_sizes = [6, 6, 6, 3]
        
        onn = HybridONN(
            layer_sizes=layer_sizes,
            hybrid_mode=HybridMode.ALTERNATING,  # C→I→C transitions
            device=device,
            transition_loss=0.15  # 15% loss de la demo
        )
        
        # Test que el forward pass funciona sin NaN/Inf
        x = torch.randn(8, 6, device=device) * 1.0
        y = onn(x)
        
        # Verificaciones básicas de sanity
        assert y.shape == (8, 3), f"Wrong output shape: {y.shape}"
        assert not torch.any(torch.isnan(y)), "Output contains NaN"
        assert not torch.any(torch.isinf(y)), "Output contains Inf"
        
        # Test que la red produce outputs diferentes para inputs diferentes
        x2 = torch.randn(8, 6, device=device) * 1.0
        y2 = onn(x2)
        
        output_diff = torch.norm(y - y2)
        assert output_diff > 1e-6, "Network not responding to different inputs"
        
        # Test que las transiciones están siendo detectadas correctamente
        metrics = onn.get_hybrid_metrics()
        expected_transitions = 2  # C→I→C para ALTERNATING mode
        actual_transitions = metrics["transition_analysis"]["total_transitions"]
        assert actual_transitions == expected_transitions, f"Wrong transition count: {actual_transitions} != {expected_transitions}"
        
        # Validar física general
        physics = onn.validate_hybrid_physics(verbose=False)
        assert physics["overall_valid"] == True
        assert physics["checks"]["transitions"]["valid"] == True
        
        print(f"   ✅ Forward pass working: {x.shape} → {y.shape}")
        print(f"   ✅ Transitions detected: {actual_transitions}")
        print(f"   ✅ Output diversity: {output_diff:.4f}")
        print(f"   ✅ Physics validation passed")
    
    @pytest.mark.skipif(not HYBRID_AVAILABLE, reason="HybridONN not available")
    def test_use_case_factories_from_demo(self, device):
        """Test factory functions para casos de uso específicos de la demo."""
        
        # Import factory functions
        from torchonn.onns.architectures.hybrid_onn import (
            create_image_processing_hybrid,
            create_signal_processing_hybrid,
            create_large_scale_hybrid
        )
        
        # Test Image Processing (TAMAÑOS PEQUEÑOS PARA TEST)
        print("   Testing Image Processing factory...")
        img_size = 8  # 8x8 en lugar de 32x32 para test rápido
        n_classes = 10
        img_onn = create_image_processing_hybrid(input_size=img_size * img_size, n_classes=n_classes)
        assert img_onn.hybrid_mode == HybridMode.FRONT_COHERENT
        assert img_onn.layer_sizes[-1] == n_classes
        
        # Test forward pass
        x_img = torch.randn(4, img_size * img_size, device=device) * 0.5
        y_img = img_onn(x_img)
        assert y_img.shape == (4, n_classes)
        print(f"     ✅ Image ONN: {x_img.shape} → {y_img.shape}")
        
        # Test Signal Processing (TAMAÑOS PEQUEÑOS PARA TEST)
        print("   Testing Signal Processing factory...")
        sig_input_size = 32  # Tamaño pequeño para test
        sig_output_size = 8
        sig_onn = create_signal_processing_hybrid(input_size=sig_input_size, output_size=sig_output_size)
        assert sig_onn.hybrid_mode == HybridMode.ALTERNATING
        assert sig_onn.n_wavelengths == 8  # Para WDM parallelism
        
        x_sig = torch.randn(4, sig_input_size, device=device) * 0.5
        y_sig = sig_onn(x_sig)
        assert y_sig.shape == (4, sig_output_size)
        print(f"     ✅ Signal ONN: {x_sig.shape} → {y_sig.shape}")
        
        # Test Large Scale (TAMAÑOS MUCHO MÁS PEQUEÑOS PARA TEST)
        print("   Testing Large Scale factory...")
        small_layer_sizes = [32, 16, 8, 4]  # Mucho más pequeño que [1024, 512, 256, 64]
        large_onn = create_large_scale_hybrid(layer_sizes=small_layer_sizes)
        assert large_onn.hybrid_mode == HybridMode.ADAPTIVE
        assert large_onn.transition_loss == 0.1  # Optimized coupling
        
        x_large = torch.randn(4, small_layer_sizes[0], device=device) * 0.5
        y_large = large_onn(x_large)
        assert y_large.shape == (4, small_layer_sizes[-1])
        print(f"     ✅ Large Scale ONN: {x_large.shape} → {y_large.shape}")
        
        print("   ✅ All factory functions working correctly")
    
    @pytest.mark.skipif(not HYBRID_AVAILABLE, reason="HybridONN not available")
    def test_wavelength_scaling_from_demo(self, device):
        """Test escalabilidad WDM validada en la demo."""
        
        layer_sizes = [8, 8, 4]
        wavelength_counts = [1, 2, 4, 8]
        
        for n_wl in wavelength_counts:
            onn = HybridONN(
                layer_sizes=layer_sizes,
                hybrid_mode=HybridMode.ALTERNATING,
                n_wavelengths=n_wl,
                device=device
            )
            
            x = torch.randn(16, 8, device=device) * 0.5
            y = onn(x)
            
            # Validar output
            assert y.shape == (16, 4)
            assert not torch.any(torch.isnan(y))
            
            # Métricas
            metrics = onn.get_hybrid_metrics()
            theoretical_speedup = metrics["resource_utilization"]["theoretical_speedup"]
            assert theoretical_speedup >= 1.0
    
    @pytest.mark.skipif(not HYBRID_AVAILABLE, reason="HybridONN not available")
    def test_training_convergence(self, device):
        """Test convergencia en entrenamiento validada en la demo."""
        
        n_samples = 200
        layer_sizes = [6, 8, 3]
        
        # Datos sintéticos
        X = torch.randn(n_samples, layer_sizes[0], device=device)
        y_target = torch.randint(0, layer_sizes[-1], (n_samples,), device=device)
        
        # Test múltiples configuraciones híbridas
        modes_to_test = [HybridMode.ALTERNATING, HybridMode.FRONT_COHERENT]
        
        for mode in modes_to_test:
            onn = HybridONN(
                layer_sizes=layer_sizes,
                hybrid_mode=mode,
                device=device
            )
            
            # Setup training
            optimizer = torch.optim.Adam(onn.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            initial_loss = None
            final_loss = None
            
            # Entrenamiento corto
            for epoch in range(20):
                optimizer.zero_grad()
                outputs = onn(X)
                loss = criterion(outputs, y_target)
                loss.backward()
                optimizer.step()
                
                if epoch == 0:
                    initial_loss = loss.item()
                elif epoch == 19:
                    final_loss = loss.item()
            
            # Verificar convergencia
            improvement = (initial_loss - final_loss) / initial_loss
            assert improvement > 0.05, f"{mode.value} didn't converge sufficiently"
    
    @pytest.mark.skipif(not HYBRID_AVAILABLE, reason="HybridONN not available")
    def test_integration_with_existing_onns(self, device, layer_sizes):
        """Test integración con CoherentONN e IncoherentONN existentes."""
        
        # Import arquitecturas existentes
        from torchonn.onns.architectures import CoherentONN, IncoherentONN
        
        # Crear las tres arquitecturas
        coherent_onn = CoherentONN(layer_sizes=layer_sizes, device=device)
        incoherent_onn = IncoherentONN(layer_sizes=layer_sizes, n_wavelengths=4, device=device)
        
        # HybridONN en modos puros debería aproximar las arquitecturas puras
        pure_coherent_hybrid = HybridONN(
            layer_sizes=layer_sizes,
            hybrid_mode=HybridMode.PURE_COHERENT,
            device=device
        )
        
        pure_incoherent_hybrid = HybridONN(
            layer_sizes=layer_sizes,
            hybrid_mode=HybridMode.PURE_INCOHERENT,
            n_wavelengths=4,
            device=device
        )
        
        # Test mismo input
        x = torch.randn(4, layer_sizes[0], device=device) * 0.5
        
        # Forward passes
        y_coherent = coherent_onn(x)
        y_incoherent = incoherent_onn(x)
        y_pure_coherent_hybrid = pure_coherent_hybrid(x)
        y_pure_incoherent_hybrid = pure_incoherent_hybrid(x)
        
        # Verificar shapes
        expected_shape = (4, layer_sizes[-1])
        assert y_coherent.shape == expected_shape
        assert y_incoherent.shape == expected_shape  
        assert y_pure_coherent_hybrid.shape == expected_shape
        assert y_pure_incoherent_hybrid.shape == expected_shape


def run_all_hybrid_tests():
    """Ejecutar todos los tests de HybridONN manualmente."""
    
    if not HYBRID_AVAILABLE:
        print("❌ HybridONN not available - cannot run tests")
        return False
    
    print("🧪 Running HybridONN Test Suite")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_sizes = [8, 12, 8, 4]
    
    try:
        tester = TestHybridONN()
        
        print("🔬 Testing hybrid modes...")
        tester.test_hybrid_modes_from_demo(device, layer_sizes)
        
        print("🔬 Testing transition physics...")
        tester.test_transition_physics_from_demo(device)
        
        print("🔬 Testing use case factories...")
        tester.test_use_case_factories_from_demo(device)
        
        print("🔬 Testing wavelength scaling...")
        tester.test_wavelength_scaling_from_demo(device)
        
        print("🔬 Testing training convergence...")
        tester.test_training_convergence(device)
        
        print("🔬 Testing integration...")
        tester.test_integration_with_existing_onns(device, layer_sizes)
        
        print("\n🎉 ALL HYBRID TESTS PASSED!")
        print("✅ HybridONN implementation fully validated")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_hybrid_tests()
    exit(0 if success else 1)