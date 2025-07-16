"""
Tests for layers module
"""

import pytest
import torch
import numpy as np
from torchonn.layers import MZILayer, MZIBlockLinear

class TestMZILayer:
    """Test MZI Layer functionality."""
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = MZILayer(in_features=10, out_features=5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight.shape == (5, 10)
    
    def test_forward_pass(self):
        """Test forward pass."""
        layer = MZILayer(in_features=10, out_features=5)
        x = torch.randn(3, 10)
        output = layer(x)
        assert output.shape == (3, 5)
    
    def test_different_devices(self):
        """Test with different devices."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = MZILayer(in_features=10, out_features=5, device=device)
        x = torch.randn(3, 10, device=device)
        output = layer(x)
        assert output.device == device
    
    def test_parameter_reset(self):
        """Test parameter reset."""
        layer = MZILayer(in_features=10, out_features=5)
        original_weight = layer.weight.clone()
        layer.reset_parameters()
        assert not torch.equal(layer.weight, original_weight)

class TestMZIBlockLinear:
    """Test MZI Block Linear functionality."""
    
    def test_initialization_usv_mode(self):
        """Test initialization in USV mode."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="usv"
        )
        assert layer.in_features == 8
        assert layer.out_features == 4
        assert layer.miniblock == 2
        assert layer.mode == "usv"
        assert hasattr(layer, 'u_matrix')
        assert hasattr(layer, 's_matrix')
        assert hasattr(layer, 'v_matrix')
    
    def test_initialization_weight_mode(self):
        """Test initialization in weight mode."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="weight"
        )
        assert layer.mode == "weight"
        assert hasattr(layer, 'weight')
        assert layer.weight.shape == (4, 8)
    
    def test_initialization_phase_mode(self):
        """Test initialization in phase mode."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="phase"
        )
        assert layer.mode == "phase"
        assert hasattr(layer, 'phases')
    
    def test_forward_pass_usv(self):
        """Test forward pass in USV mode."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="usv"
        )
        x = torch.randn(2, 8)
        output = layer(x)
        assert output.shape == (2, 4)
    
    def test_forward_pass_weight(self):
        """Test forward pass in weight mode."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="weight"
        )
        x = torch.randn(2, 8)
        output = layer(x)
        assert output.shape == (2, 4)
    
    def test_forward_pass_phase(self):
        """Test forward pass in phase mode."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="phase"
        )
        x = torch.randn(2, 8)
        output = layer(x)
        assert output.shape == (2, 4)
    
    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            MZIBlockLinear(
                in_features=8, 
                out_features=4, 
                miniblock=2, 
                mode="invalid"
            )
    
    def test_invalid_dimensions(self):
        """Test invalid dimensions raise error."""
        with pytest.raises(ValueError):
            MZIBlockLinear(
                in_features=0, 
                out_features=4, 
                miniblock=2, 
                mode="usv"
            )
        
        with pytest.raises(ValueError):
            MZIBlockLinear(
                in_features=8, 
                out_features=-1, 
                miniblock=2, 
                mode="usv"
            )
    
    def test_parameter_reset(self):
        """Test parameter reset."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="usv"
        )
        original_u = layer.u_matrix.clone()
        layer.reset_parameters()
        assert not torch.equal(layer.u_matrix, original_u)
    
    def test_get_weight_matrix(self):
        """Test weight matrix generation."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="usv"
        )
        weight = layer._get_weight_matrix()
        assert weight.shape == (4, 8)
        assert weight.dtype == layer.dtype
    
    def test_input_size_mismatch(self):
        """Test input size mismatch raises error."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="usv"
        )
        x = torch.randn(2, 10)  # Wrong input size
        with pytest.raises(ValueError):
            layer(x)
    
    def test_gradient_flow(self):
        """Test gradient flow through layer."""
        layer = MZIBlockLinear(
            in_features=8, 
            out_features=4, 
            miniblock=2, 
            mode="usv"
        )
        x = torch.randn(2, 8, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        for param in layer.parameters():
            assert param.grad is not None