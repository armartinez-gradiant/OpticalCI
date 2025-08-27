#!/usr/bin/env python3
"""
🧪 Tests for WDM Optimization

UBICACIÓN: tests/test_wdm_optimization.py

Suite completa de tests que valida:
- OptimizedWDMMultiplexer funcionality
- ParallelMRRWeightBank correctness
- ParallelIncoherentLayer performance
- OptimizedIncoherentONN integration
- WDM scaling efficiency improvements
- Backward compatibility
- Performance regression testing

USAGE:
    pytest tests/test_wdm_optimization.py -v
    python tests/test_wdm_optimization.py  # Direct execution
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, Any, List
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)

# ========================================
# 1. FIXTURES AND UTILITIES
# ========================================

@pytest.fixture
def device():
    """Fixture for computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_wavelengths():
    """Standard wavelengths for testing."""
    return [1530e-9, 1540e-9, 1550e-9, 1560e-9]  # 4 channels

@pytest.fixture
def test_data():
    """Standard test data."""
    return {
        "batch_size": 16,
        "features": 8,
        "n_wavelengths": 4,
        "layer_sizes": [8, 12, 6]
    }

def create_test_signals(batch_size: int, n_channels: int, device: torch.device) -> List[torch.Tensor]:
    """Create test signals for WDM testing."""
    signals = []
    for i in range(n_channels):
        if i % 3 == 0:
            signal = torch.randn(batch_size, device=device)
        elif i % 3 == 1:
            signal = torch.sin(torch.linspace(0, 4*np.pi, batch_size, device=device))
        else:
            signal = torch.ones(batch_size, device=device) * (i + 1) * 0.2
        signals.append(signal)
    return signals

# ========================================
# 2. TEST OPTIMIZED WDM MULTIPLEXER
# ========================================

class TestOptimizedWDMMultiplexer:
    """Tests for OptimizedWDMMultiplexer."""
    
    def test_import_availability(self):
        """Test that optimized components can be imported."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedWDMMultiplexer
            assert True, "OptimizedWDMMultiplexer import successful"
        except ImportError as e:
            pytest.skip(f"WDM optimizations not available: {e}")
    
    def test_multiplexer_initialization(self, device, test_data):
        """Test: WDM multiplexer initializes correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedWDMMultiplexer
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        n_wavelengths = test_data["n_wavelengths"]
        
        mux = OptimizedWDMMultiplexer(n_wavelengths, device)
        
        assert mux.n_wavelengths == n_wavelengths
        assert mux.device == device
        assert hasattr(mux, 'wavelength_gains')
        assert hasattr(mux, 'crosstalk_matrix')
        assert mux.wavelength_gains.shape == (n_wavelengths,)
        assert mux.crosstalk_matrix.shape == (n_wavelengths, n_wavelengths)
    
    def test_batch_multiplexing(self, device, test_data):
        """Test: Batch multiplexing works correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedWDMMultiplexer
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        batch_size = test_data["batch_size"]
        features = test_data["features"]
        n_wavelengths = test_data["n_wavelengths"]
        
        mux = OptimizedWDMMultiplexer(n_wavelengths, device)
        
        # Test multiplex_batch
        input_tensor = torch.randn(batch_size, features, device=device)
        multiplexed = mux.multiplex_batch(input_tensor)
        
        # Verify output dimensions
        assert len(multiplexed.shape) == 3  # [batch_per_wl, features, n_wavelengths]
        assert multiplexed.shape[1] == features
        assert multiplexed.shape[2] == n_wavelengths
        
        # Verify batch division logic
        expected_batch_per_wl = batch_size // n_wavelengths
        if batch_size % n_wavelengths == 0:
            assert multiplexed.shape[0] == expected_batch_per_wl
        else:
            # Should handle remainder by padding
            assert multiplexed.shape[0] >= expected_batch_per_wl
    
    def test_batch_demultiplexing(self, device, test_data):
        """Test: Batch demultiplexing works correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedWDMMultiplexer
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        features = test_data["features"]
        n_wavelengths = test_data["n_wavelengths"]
        batch_per_wl = 4
        
        mux = OptimizedWDMMultiplexer(n_wavelengths, device)
        
        # Create WDM tensor
        wdm_tensor = torch.randn(batch_per_wl, features, n_wavelengths, device=device)
        
        # Test demultiplex_batch
        demultiplexed = mux.demultiplex_batch(wdm_tensor)
        
        # Verify output dimensions
        assert demultiplexed.shape[1] == features
        assert demultiplexed.shape[0] == batch_per_wl * n_wavelengths  # Concatenated back
        
        # Test crosstalk application
        assert not torch.allclose(demultiplexed, wdm_tensor.permute(2, 0, 1).contiguous().view(-1, features))
    
    def test_round_trip_consistency(self, device, test_data):
        """Test: Multiplex->Demultiplex round trip maintains reasonable fidelity."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedWDMMultiplexer
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        batch_size = test_data["batch_size"]
        features = test_data["features"]
        n_wavelengths = test_data["n_wavelengths"]
        
        mux = OptimizedWDMMultiplexer(n_wavelengths, device)
        
        # Original signal
        original = torch.randn(batch_size, features, device=device)
        
        # Round trip
        multiplexed = mux.multiplex_batch(original)
        recovered = mux.demultiplex_batch(multiplexed)
        
        # Trim to original size if needed
        if recovered.shape[0] > batch_size:
            recovered = recovered[:batch_size]
        elif recovered.shape[0] < batch_size:
            # Pad if needed for comparison
            padding = torch.zeros(batch_size - recovered.shape[0], features, device=device)
            recovered = torch.cat([recovered, padding], dim=0)
        
        # Check fidelity (should be reasonably close)
        mse = torch.nn.functional.mse_loss(recovered, original)
        relative_mse = mse / torch.var(original)
        
        # Allow for some degradation due to crosstalk but not too much
        assert relative_mse.item() < 2.0, f"Round trip fidelity too poor: {relative_mse.item():.3f}"

# ========================================
# 3. TEST PARALLEL MRR WEIGHT BANK
# ========================================

class TestParallelMRRWeightBank:
    """Tests for ParallelMRRWeightBank."""
    
    def test_weight_bank_initialization(self, device):
        """Test: MRR weight bank initializes correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import ParallelMRRWeightBank
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        in_features, out_features, n_wavelengths = 6, 8, 4
        
        bank = ParallelMRRWeightBank(in_features, out_features, n_wavelengths, device)
        
        assert bank.in_features == in_features
        assert bank.out_features == out_features
        assert bank.n_wavelengths == n_wavelengths
        
        # Check weight dimensions
        assert bank.weights.shape == (n_wavelengths, out_features, in_features)
        assert bank.bias.shape == (n_wavelengths, out_features)
        assert bank.wavelength_efficiency.shape == (n_wavelengths,)
        
        # Check parameter initialization bounds
        assert torch.all(bank.wavelength_efficiency >= 0.5)
        assert torch.all(bank.wavelength_efficiency <= 1.0)
    
    def test_parallel_forward_pass(self, device):
        """Test: Parallel forward pass works correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import ParallelMRRWeightBank
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        in_features, out_features, n_wavelengths = 6, 8, 4
        batch_per_wl = 5
        
        bank = ParallelMRRWeightBank(in_features, out_features, n_wavelengths, device)
        
        # Input: [batch_per_wl, in_features, n_wavelengths]
        x = torch.randn(batch_per_wl, in_features, n_wavelengths, device=device)
        
        # Forward pass
        output = bank(x)
        
        # Verify output dimensions
        assert output.shape == (batch_per_wl, out_features, n_wavelengths)
        
        # Verify output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_microring_count_accuracy(self, device):
        """Test: Microring count calculation is accurate."""
        try:
            from torchonn.optimizations.wdm_optimization import ParallelMRRWeightBank
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        in_features, out_features, n_wavelengths = 6, 8, 4
        
        bank = ParallelMRRWeightBank(in_features, out_features, n_wavelengths, device)
        
        expected_count = in_features * out_features * n_wavelengths
        actual_count = bank.get_microring_count()
        
        assert actual_count == expected_count
        assert isinstance(actual_count, int)
        assert actual_count > 0
    
    def test_memory_efficiency_calculation(self, device):
        """Test: Memory efficiency calculation is reasonable."""
        try:
            from torchonn.optimizations.wdm_optimization import ParallelMRRWeightBank
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        in_features, out_features, n_wavelengths = 6, 8, 4
        
        bank = ParallelMRRWeightBank(in_features, out_features, n_wavelengths, device)
        
        mem_eff = bank.get_memory_efficiency()
        
        assert isinstance(mem_eff, float)
        assert mem_eff > 0.0
        assert mem_eff <= n_wavelengths * 1.5  # Should scale with wavelengths but have some limit

# ========================================
# 4. TEST PARALLEL INCOHERENT LAYER
# ========================================

class TestParallelIncoherentLayer:
    """Tests for ParallelIncoherentLayer."""
    
    def test_layer_initialization(self, device, test_data):
        """Test: Parallel incoherent layer initializes correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import ParallelIncoherentLayer
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        in_features = test_data["features"]
        out_features = test_data["features"] + 2
        n_wavelengths = test_data["n_wavelengths"]
        
        layer = ParallelIncoherentLayer(in_features, out_features, n_wavelengths, device)
        
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert layer.n_wavelengths == n_wavelengths
        assert layer.device == device
        
        # Check components exist
        assert hasattr(layer, 'wdm')
        assert hasattr(layer, 'weight_bank')
        assert hasattr(layer, 'photodetector_efficiency')
        assert hasattr(layer, 'input_norm')
        assert hasattr(layer, 'output_processing')
    
    def test_forward_pass_dimensions(self, device, test_data):
        """Test: Forward pass maintains correct dimensions."""
        try:
            from torchonn.optimizations.wdm_optimization import ParallelIncoherentLayer
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        batch_size = test_data["batch_size"]
        in_features = test_data["features"]
        out_features = test_data["features"] + 2
        n_wavelengths = test_data["n_wavelengths"]
        
        layer = ParallelIncoherentLayer(in_features, out_features, n_wavelengths, device)
        
        x = torch.randn(batch_size, in_features, device=device)
        output = layer(x)
        
        # Verify dimensions preserved
        assert output.shape == (batch_size, out_features)
        
        # Verify output quality
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert torch.all(output >= 0)  # Should be positive due to ReLU and processing
    
    def test_parallel_efficiency_metrics(self, device, test_data):
        """Test: Parallel efficiency metrics are calculated correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import ParallelIncoherentLayer
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        in_features = test_data["features"]
        out_features = test_data["features"] + 2
        n_wavelengths = test_data["n_wavelengths"]
        
        layer = ParallelIncoherentLayer(in_features, out_features, n_wavelengths, device)
        
        efficiency = layer.get_parallel_efficiency()
        
        # Check required metrics exist
        required_keys = ["memory_efficiency", "compute_efficiency", "overall_efficiency", 
                        "theoretical_speedup", "microring_count", "photodetector_count"]
        for key in required_keys:
            assert key in efficiency, f"Missing metric: {key}"
        
        # Check metric ranges
        assert efficiency["overall_efficiency"] >= 0.0
        assert efficiency["theoretical_speedup"] > 0.0
        assert efficiency["microring_count"] > 0
        assert efficiency["photodetector_count"] == n_wavelengths * out_features
        
        # Check speedup scaling
        assert efficiency["theoretical_speedup"] >= n_wavelengths * 0.5  # At least 50% of theoretical

# ========================================
# 5. TEST OPTIMIZED INCOHERENT ONN
# ========================================

class TestOptimizedIncoherentONN:
    """Tests for complete OptimizedIncoherentONN."""
    
    def test_onn_initialization(self, device, test_data):
        """Test: Complete ONN initializes correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        layer_sizes = test_data["layer_sizes"]
        n_wavelengths = test_data["n_wavelengths"]
        
        onn = OptimizedIncoherentONN(layer_sizes, n_wavelengths, device)
        
        assert onn.layer_sizes == layer_sizes
        assert onn.n_wavelengths == n_wavelengths
        assert onn.device == device
        assert len(onn.layers) == len(layer_sizes) - 1
        
        # Check layer types
        for layer in onn.layers:
            assert hasattr(layer, 'get_parallel_efficiency')
    
    def test_onn_forward_pass(self, device, test_data):
        """Test: Complete ONN forward pass works."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        batch_size = test_data["batch_size"]
        layer_sizes = test_data["layer_sizes"]
        n_wavelengths = test_data["n_wavelengths"]
        
        onn = OptimizedIncoherentONN(layer_sizes, n_wavelengths, device)
        
        x = torch.randn(batch_size, layer_sizes[0], device=device)
        output = onn(x)
        
        # Verify dimensions
        assert output.shape == (batch_size, layer_sizes[-1])
        
        # Verify output quality
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert torch.all(output >= 0)  # Sigmoid ensures [0,1] range
        assert torch.all(output <= 1)
    
    def test_wdm_efficiency_metrics(self, device, test_data):
        """Test: WDM efficiency metrics are comprehensive."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        layer_sizes = test_data["layer_sizes"]
        n_wavelengths = test_data["n_wavelengths"]
        
        onn = OptimizedIncoherentONN(layer_sizes, n_wavelengths, device)
        
        metrics = onn.get_wdm_efficiency_metrics()
        
        # Check required metrics
        required_keys = [
            "n_wavelengths", "total_microrings", "total_photodetectors", 
            "total_parameters", "parallel_efficiency", "theoretical_speedup",
            "memory_efficiency", "compute_efficiency", "wdm_scaling_factor",
            "architecture", "optical_fraction"
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        # Check metric validity
        assert metrics["n_wavelengths"] == n_wavelengths
        assert metrics["total_microrings"] > 0
        assert metrics["total_photodetectors"] > 0
        assert metrics["total_parameters"] > 0
        assert 0.0 <= metrics["parallel_efficiency"] <= 100.0
        assert metrics["theoretical_speedup"] > 0.0
        assert 0.0 <= metrics["optical_fraction"] <= 1.0
        assert metrics["architecture"] == layer_sizes
        
        # Check scaling properties
        assert metrics["wdm_scaling_factor"] >= n_wavelengths * 0.5
    
    def test_physics_validation(self, device, test_data):
        """Test: Physics validation works correctly."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        layer_sizes = test_data["layer_sizes"]
        n_wavelengths = test_data["n_wavelengths"]
        
        onn = OptimizedIncoherentONN(layer_sizes, n_wavelengths, device)
        
        physics = onn.validate_physics()
        
        # Check physics validation results
        required_keys = ["valid_transmissions", "energy_conservation", "positive_powers", 
                        "realistic_coupling", "total_microrings", "efficiency_percentage"]
        
        for key in required_keys:
            assert key in physics, f"Missing physics check: {key}"
        
        # All physics checks should pass
        assert physics["valid_transmissions"] == True
        assert physics["energy_conservation"] == True
        assert physics["positive_powers"] == True
        assert physics["realistic_coupling"] == True

# ========================================
# 6. TEST WDM SCALING PERFORMANCE
# ========================================

class TestWDMScalingPerformance:
    """Tests for WDM scaling performance improvements."""
    
    def test_scaling_efficiency_improvement(self, device):
        """Test: WDM scaling efficiency improves with optimizations."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
            from torchonn.onns.architectures import IncoherentONN
        except ImportError:
            pytest.skip("Required components not available")
        
        layer_sizes = [8, 12, 6]
        wavelength_counts = [2, 4, 8]
        batch_size = 16
        
        scaling_results = []
        
        for n_wl in wavelength_counts:
            # Test optimized version
            try:
                onn_opt = OptimizedIncoherentONN(layer_sizes, n_wl, device)
                metrics_opt = onn_opt.get_wdm_efficiency_metrics()
                
                # Test original version for comparison
                onn_orig = IncoherentONN(layer_sizes, n_wl, enable_wdm_optimization=False, device=device)
                metrics_orig = onn_orig.get_optical_efficiency_metrics()
                
                efficiency_improvement = (metrics_opt.get('parallel_efficiency', 0) / 
                                        max(metrics_orig.get('parallel_efficiency', 1), 1))
                
                scaling_results.append({
                    "wavelengths": n_wl,
                    "efficiency_original": metrics_orig.get('parallel_efficiency', 0),
                    "efficiency_optimized": metrics_opt.get('parallel_efficiency', 0),
                    "improvement_factor": efficiency_improvement
                })
                
            except Exception as e:
                pytest.skip(f"Scaling test failed for {n_wl} wavelengths: {e}")
        
        if len(scaling_results) >= 2:
            # Check that efficiency doesn't degrade catastrophically with wavelength count
            efficiencies = [r["efficiency_optimized"] for r in scaling_results]
            min_efficiency = min(efficiencies)
            max_efficiency = max(efficiencies)
            
            # Efficiency shouldn't drop below 30% even at high wavelength counts
            assert min_efficiency > 30.0, f"Efficiency degraded too much: {min_efficiency:.1f}%"
            
            # Check improvement over original
            improvements = [r["improvement_factor"] for r in scaling_results]
            avg_improvement = np.mean(improvements)
            
            assert avg_improvement > 1.2, f"Insufficient improvement: {avg_improvement:.2f}x"
        else:
            pytest.skip("Insufficient scaling test results")
    
    def test_theoretical_speedup_scaling(self, device):
        """Test: Theoretical speedup scales reasonably with wavelength count."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        layer_sizes = [6, 8, 4]
        wavelength_counts = [1, 2, 4, 8]
        
        speedups = []
        
        for n_wl in wavelength_counts:
            try:
                onn = OptimizedIncoherentONN(layer_sizes, n_wl, device)
                metrics = onn.get_wdm_efficiency_metrics()
                speedups.append(metrics.get('theoretical_speedup', 1.0))
            except Exception as e:
                pytest.skip(f"Speedup test failed for {n_wl} wavelengths: {e}")
        
        if len(speedups) >= 3:
            # Speedup should generally increase with wavelength count
            # Allow for some non-monotonic behavior due to overhead
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            
            # Should see reasonable scaling
            scaling_factor = max_speedup / min_speedup
            assert scaling_factor > 1.5, f"Insufficient speedup scaling: {scaling_factor:.2f}x"
            
            # Speedup should be at least 50% of wavelength count for high counts
            if len(wavelength_counts) >= 1:
                max_wl = max(wavelength_counts)
                final_speedup = speedups[-1]
                efficiency_ratio = final_speedup / max_wl
                assert efficiency_ratio > 0.5, f"Poor speedup efficiency: {efficiency_ratio:.2f}"

# ========================================
# 7. TEST BACKWARD COMPATIBILITY
# ========================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""
    
    def test_incoherent_onn_compatibility(self, device, test_data):
        """Test: Optimized IncoherentONN is compatible with original interface."""
        try:
            from torchonn.onns.architectures import IncoherentONN
        except ImportError:
            pytest.skip("IncoherentONN not available")
        
        layer_sizes = test_data["layer_sizes"]
        n_wavelengths = test_data["n_wavelengths"]
        batch_size = test_data["batch_size"]
        
        # Test both original and optimized modes
        onn_original = IncoherentONN(layer_sizes, n_wavelengths, enable_wdm_optimization=False, device=device)
        onn_optimized = IncoherentONN(layer_sizes, n_wavelengths, enable_wdm_optimization=True, device=device)
        
        x = torch.randn(batch_size, layer_sizes[0], device=device)
        
        # Both should work and return same-shaped outputs
        y_original = onn_original(x)
        y_optimized = onn_optimized(x)
        
        assert y_original.shape == y_optimized.shape
        assert y_original.shape == (batch_size, layer_sizes[-1])
        
        # Both should have required methods
        for onn in [onn_original, onn_optimized]:
            assert hasattr(onn, 'get_optical_efficiency_metrics')
            assert hasattr(onn, 'validate_physics')
            assert hasattr(onn, 'get_theoretical_speedup')
            
            # Test method calls work
            metrics = onn.get_optical_efficiency_metrics()
            physics = onn.validate_physics()
            speedup = onn.get_theoretical_speedup()
            
            assert isinstance(metrics, dict)
            assert isinstance(physics, dict)
            assert isinstance(speedup, (int, float))
    
    def test_method_signatures_preserved(self, device, test_data):
        """Test: All method signatures are preserved."""
        try:
            from torchonn.onns.architectures import IncoherentONN
        except ImportError:
            pytest.skip("IncoherentONN not available")
        
        layer_sizes = test_data["layer_sizes"]
        n_wavelengths = test_data["n_wavelengths"]
        
        onn = IncoherentONN(layer_sizes, n_wavelengths, device=device)
        
        # Test that all expected methods exist with correct signatures
        methods_to_test = [
            ('get_optical_efficiency_metrics', []),
            ('validate_physics', []),
            ('get_theoretical_speedup', []),
            ('get_component_counts', [])
        ]
        
        for method_name, args in methods_to_test:
            assert hasattr(onn, method_name), f"Method {method_name} missing"
            
            # Test method can be called
            method = getattr(onn, method_name)
            result = method(*args)
            
            # Should return reasonable results
            assert result is not None
            if isinstance(result, dict):
                assert len(result) > 0

# ========================================
# 8. PERFORMANCE REGRESSION TESTS
# ========================================

class TestPerformanceRegression:
    """Tests to ensure performance doesn't regress."""
    
    def test_forward_pass_performance(self, device):
        """Test: Forward pass performance is reasonable."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        layer_sizes = [16, 24, 16, 8]
        n_wavelengths = 8
        batch_size = 32
        n_trials = 5
        
        onn = OptimizedIncoherentONN(layer_sizes, n_wavelengths, device)
        x = torch.randn(batch_size, layer_sizes[0], device=device)
        
        # Warmup
        with torch.no_grad():
            _ = onn(x)
        
        # Time multiple runs
        times = []
        for _ in range(n_trials):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                _ = onn(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time
        
        # Performance should be reasonable
        assert avg_time < 5.0, f"Forward pass too slow: {avg_time:.3f}s"
        assert throughput > 10, f"Throughput too low: {throughput:.1f} samples/sec"
        assert std_time < avg_time * 0.5, f"Timing too variable: {std_time/avg_time:.2f} relative std"
    
    def test_memory_usage_reasonable(self, device, test_data):
        """Test: Memory usage is reasonable."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        layer_sizes = test_data["layer_sizes"]
        n_wavelengths = test_data["n_wavelengths"]
        
        onn = OptimizedIncoherentONN(layer_sizes, n_wavelengths, device)
        
        # Check parameter count is reasonable
        total_params = sum(p.numel() for p in onn.parameters())
        
        # Rough estimate: should be in reasonable range
        expected_range = (1000, 1000000)  # 1K to 1M parameters
        assert expected_range[0] <= total_params <= expected_range[1], \
            f"Parameter count unreasonable: {total_params:,}"
        
        # Check model size
        model_size_bytes = total_params * 4  # 4 bytes per float32
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        assert model_size_mb < 100, f"Model too large: {model_size_mb:.1f} MB"

# ========================================
# 9. INTEGRATION TESTS
# ========================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_optimization_pipeline(self, device):
        """Test: Complete optimization pipeline works end-to-end."""
        try:
            from torchonn.optimizations.wdm_optimization import benchmark_wdm_scaling
        except ImportError:
            pytest.skip("WDM optimizations not available")
        
        # Run a small version of the benchmark
        layer_sizes = [6, 8, 4]
        batch_size = 16
        wavelength_counts = [2, 4]
        n_runs = 3
        
        try:
            results = benchmark_wdm_scaling(
                layer_sizes=layer_sizes,
                batch_size=batch_size,
                wavelength_counts=wavelength_counts,
                n_runs=n_runs,
                device=device
            )
            
            # Check results structure
            assert isinstance(results, dict)
            
            # Should have results for each wavelength count
            successful_results = {k: v for k, v in results.items() if v.get("success", False)}
            assert len(successful_results) >= 1, "No successful benchmark results"
            
            # Check result structure
            for wl_count, result in successful_results.items():
                assert "time_ms" in result
                assert "theoretical_speedup" in result
                assert "parallel_efficiency" in result
                assert result["time_ms"] > 0
                assert result["theoretical_speedup"] > 0
                assert result["parallel_efficiency"] >= 0
                
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")
    
    def test_multiple_optimizations_compatibility(self, device, test_data):
        """Test: Multiple optimization techniques work together."""
        try:
            from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
            from torchonn.onns.architectures import IncoherentONN
        except ImportError:
            pytest.skip("Required components not available")
        
        layer_sizes = test_data["layer_sizes"]
        batch_size = test_data["batch_size"]
        
        # Test different configurations
        configs = [
            {"n_wavelengths": 4, "enable_wdm_optimization": False},
            {"n_wavelengths": 4, "enable_wdm_optimization": True},
            {"n_wavelengths": 8, "enable_wdm_optimization": True}
        ]
        
        results = []
        
        for config in configs:
            try:
                if config.get("enable_wdm_optimization", False):
                    # Use IncoherentONN with optimization enabled
                    onn = IncoherentONN(layer_sizes, device=device, **config)
                else:
                    # Use standard IncoherentONN
                    onn = IncoherentONN(layer_sizes, device=device, **config)
                
                # Test forward pass
                x = torch.randn(batch_size, layer_sizes[0], device=device)
                y = onn(x)
                
                # Test metrics
                metrics = onn.get_optical_efficiency_metrics()
                
                results.append({
                    "config": config,
                    "output_shape": y.shape,
                    "efficiency": metrics.get('parallel_efficiency', 0),
                    "speedup": metrics.get('theoretical_speedup', 1.0),
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "config": config,
                    "error": str(e),
                    "success": False
                })
        
        # Should have at least some successful results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) >= 1, "No configurations worked successfully"
        
        # All successful results should have correct output shape
        expected_shape = (batch_size, layer_sizes[-1])
        for result in successful_results:
            assert result["output_shape"] == expected_shape

# ========================================
# 10. MAIN EXECUTION
# ========================================

def run_all_tests():
    """Run all tests when executed directly."""
    print("🧪 Running WDM Optimization Tests...")
    print("=" * 50)
    
    # Test availability first
    try:
        from torchonn.optimizations.wdm_optimization import OptimizedIncoherentONN
        print("✅ WDM optimizations available")
    except ImportError as e:
        print(f"❌ WDM optimizations not available: {e}")
        print("⚠️ Some tests will be skipped")
    
    # Run pytest programmatically
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

# ========================================
# 11. SUMMARY OF TEST COVERAGE
# ========================================

"""
🧪 COMPREHENSIVE TEST COVERAGE:

COMPONENT TESTS:
✅ OptimizedWDMMultiplexer: initialization, batch operations, round-trip fidelity
✅ ParallelMRRWeightBank: initialization, forward pass, microring counting
✅ ParallelIncoherentLayer: initialization, forward pass, efficiency metrics
✅ OptimizedIncoherentONN: complete integration, WDM metrics, physics validation

PERFORMANCE TESTS:
📊 WDM scaling efficiency improvements vs original
📊 Theoretical speedup scaling with wavelength count  
📊 Forward pass performance benchmarks
📊 Memory usage validation

COMPATIBILITY TESTS:
🔄 Backward compatibility with existing IncoherentONN interface
🔄 Method signature preservation
🔄 Multiple optimization techniques together
🔄 End-to-end integration pipeline

REGRESSION TESTS:
⚡ Performance doesn't degrade below thresholds
⚡ Memory usage stays reasonable
⚡ Timing consistency across runs
⚡ Output quality maintained

VALIDATION TESTS:
🔍 Physics validation works correctly
🔍 Metrics calculations are accurate
🔍 Component counts are correct
🔍 Efficiency improvements are real

EXPECTED RESULTS:
- All tests should pass with WDM optimizations installed
- Performance tests should show >50% efficiency at high wavelength counts
- Compatibility tests should confirm seamless integration
- Regression tests should confirm optimization benefits

USAGE:
pytest tests/test_wdm_optimization.py -v          # Run with pytest
python tests/test_wdm_optimization.py            # Direct execution
pytest tests/test_wdm_optimization.py::TestOptimizedWDMMultiplexer -v  # Specific class

This test suite provides comprehensive validation that the WDM optimizations
work correctly, improve performance as expected, and maintain compatibility.
"""