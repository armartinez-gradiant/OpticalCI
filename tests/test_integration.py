"""
Integration tests for PtONN-TESTS

End-to-end tests that verify the complete functionality.
"""

import pytest
import torch
import numpy as np
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.models import ONNBaseModel
from torchonn.devices import DeviceConfig, get_default_device
from torchonn.ops import matrix_decomposition, apply_noise
from torchonn.utils import check_torch_version, validate_tensor_shape

class CompleteONNModel(ONNBaseModel):
    """Complete ONN model for integration testing."""
    
    def __init__(self, input_size=16, hidden_sizes=[12, 8], output_size=4, device=None):
        super().__init__(device=device)
        
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layer = MZIBlockLinear(
                in_features=prev_size,
                out_features=hidden_size,
                miniblock=4,
                mode="usv",
                device=self.device
            )
            self.layers.append(layer)
            prev_size = hidden_size
        
        # Output layer
        output_layer = MZILayer(
            in_features=prev_size,
            out_features=output_size,
            device=self.device
        )
        self.layers.append(output_layer)
        
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if self.training:
                x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

class TestCompleteIntegration:
    """Complete integration tests."""
    
    def test_environment_setup(self):
        """Test that environment is properly set up."""
        version_info = check_torch_version()
        assert version_info["version_compatible"]
        assert "torch_version" in version_info
    
    def test_device_configuration(self):
        """Test device configuration system."""
        default_config = get_default_device()
        assert isinstance(default_config, DeviceConfig)
        assert isinstance(default_config.device, torch.device)
    
    def test_complete_model_workflow(self):
        """Test complete model training workflow."""
        # Create model
        model = CompleteONNModel(input_size=16, hidden_sizes=[12, 8], output_size=4)
        
        # Create data
        batch_size = 8
        x = torch.randn(batch_size, 16)
        y = torch.randint(0, 4, (batch_size,))
        
        # Forward pass
        output = model(x)
        assert output.shape == (batch_size, 4)
        
        # Loss computation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, y)
        assert loss.item() > 0
        
        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify gradients were computed
        for param in model.parameters():
            assert param.grad is not None
    
    def test_batch_processing(self):
        """Test processing of different batch sizes."""
        model = CompleteONNModel()
        
        # Test different batch sizes
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 16)
            output = model(x)
            assert output.shape == (batch_size, 4)
    
    def test_model_persistence(self):
        """Test saving and loading models."""
        import tempfile
        import os
        
        # Create and train model
        model = CompleteONNModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Dummy training step
        x = torch.randn(4, 16)
        y = torch.randint(0, 4, (4,))
        criterion = torch.nn.CrossEntropyLoss()
        
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f.name)
            
            # Load model
            new_model = CompleteONNModel()
            new_optimizer = torch.optim.Adam(new_model.parameters())
            
            checkpoint = torch.load(f.name)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Verify loaded model works
            new_output = new_model(x)
            assert torch.allclose(output, new_output, atol=1e-6)
            
            # Cleanup
            os.unlink(f.name)
    
    def test_noise_robustness(self):
        """Test model robustness to noise."""
        model = CompleteONNModel()
        model.eval()
        
        x = torch.randn(8, 16)
        
        # Clean output
        clean_output = model(x)
        
        # Noisy input
        noisy_x = apply_noise(x, noise_level=0.1)
        noisy_output = model(noisy_x)
        
        # Outputs should be different but not dramatically
        assert not torch.allclose(clean_output, noisy_output)
        
        # But correlation should still be reasonable
        correlation = torch.corrcoef(torch.stack([
            clean_output.flatten(), 
            noisy_output.flatten()
        ]))[0, 1]
        assert correlation > 0.5  # Should maintain some correlation
    
    def test_memory_efficiency(self):
        """Test memory usage is reasonable."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory test")
        
        device = torch.device("cuda")
        model = CompleteONNModel(device=device)
        
        # Measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Forward pass
        x = torch.randn(32, 16, device=device)
        output = model(x)
        
        # Measure peak memory
        peak_memory = torch.cuda.memory_allocated(device)
        memory_used = peak_memory - initial_memory
        
        # Memory usage should be reasonable (less than 100MB for this small model)
        assert memory_used < 100 * 1024 * 1024  # 100MB
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        model = CompleteONNModel()
        model.eval()
        
        # Test with various input ranges
        test_cases = [
            torch.randn(4, 16) * 0.01,   # Very small inputs
            torch.randn(4, 16) * 10,     # Large inputs
            torch.zeros(4, 16),          # Zero inputs
            torch.ones(4, 16),           # Constant inputs
        ]
        
        for x in test_cases:
            with torch.no_grad():
                output = model(x)
                
                # Check for NaN or Inf
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()
                
                # Check output is reasonable
                assert output.abs().max() < 1000  # Not exploding
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        model = CompleteONNModel()
        
        # Create data that might cause large gradients
        x = torch.randn(16, 16) * 10
        y = torch.randint(0, 4, (16,))
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients before clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        # Apply gradient clipping
        max_norm = 1.0
        grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Verify clipping worked
        if grad_norm_before > max_norm:
            assert abs(grad_norm_after - max_norm) < 1e-6
    
    def test_different_precisions(self):
        """Test with different floating point precisions."""
        precisions = [torch.float32, torch.float64]
        
        for dtype in precisions:
            model = CompleteONNModel()
            model = model.to(dtype=dtype)
            
            x = torch.randn(4, 16, dtype=dtype)
            output = model(x)
            
            assert output.dtype == dtype
            assert not torch.isnan(output).any()
    
    def test_model_modes(self):
        """Test different model modes (train/eval)."""
        model = CompleteONNModel()
        x = torch.randn(8, 16)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Evaluation mode  
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval)
        
        # But both should be valid
        assert not torch.isnan(output_train).any()
        assert not torch.isnan(output_eval).any()