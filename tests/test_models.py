"""
Tests for models module
"""

import pytest
import torch
import numpy as np
from torchonn.models import ONNBaseModel
from torchonn.layers import MZILayer, MZIBlockLinear

class TestONNBaseModel:
    """Test ONN Base Model functionality."""
    
    def test_initialization(self):
        """Test base model initialization."""
        model = ONNBaseModel()
        assert hasattr(model, 'device')
        assert isinstance(model.device, torch.device)
    
    def test_device_specification(self):
        """Test device specification."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ONNBaseModel(device=device)
        assert model.device == device
    
    def test_string_device(self):
        """Test string device specification."""
        model = ONNBaseModel(device="cpu")
        assert model.device == torch.device("cpu")

class SimpleONNModel(ONNBaseModel):
    """Simple ONN model for testing."""
    
    def __init__(self, input_size=10, hidden_size=8, output_size=4, device=None):
        super().__init__(device=device)
        
        self.layer1 = MZIBlockLinear(
            in_features=input_size,
            out_features=hidden_size,
            miniblock=2,
            mode="usv",
            device=self.device
        )
        
        self.layer2 = MZILayer(
            in_features=hidden_size,
            out_features=output_size,
            device=self.device
        )
        
        self.activation = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

class TestSimpleONNModel:
    """Test simple ONN model implementation."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = SimpleONNModel()
        assert isinstance(model, ONNBaseModel)
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'activation')
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = SimpleONNModel(input_size=10, hidden_size=8, output_size=4)
        x = torch.randn(3, 10)
        output = model(x)
        assert output.shape == (3, 4)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = SimpleONNModel(input_size=10, hidden_size=8, output_size=4)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable
    
    def test_gradient_flow(self):
        """Test gradient flow through model."""
        model = SimpleONNModel(input_size=10, hidden_size=8, output_size=4)
        x = torch.randn(3, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None
    
    def test_training_mode(self):
        """Test training/eval mode switching."""
        model = SimpleONNModel()
        
        # Default should be training mode
        assert model.training
        
        # Switch to eval mode
        model.eval()
        assert not model.training
        
        # Switch back to training mode
        model.train()
        assert model.training
    
    def test_device_consistency(self):
        """Test device consistency across model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleONNModel(device=device)
        
        # Check all parameters are on correct device
        for param in model.parameters():
            assert param.device == device
    
    def test_reset_parameters(self):
        """Test parameter reset functionality."""
        model = SimpleONNModel()
        
        # Get original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.clone()
        
        # Reset parameters
        model.reset_parameters()
        
        # Check parameters have changed
        for name, param in model.named_parameters():
            assert not torch.equal(param, original_params[name])
    
    def test_model_state_dict(self):
        """Test model state dict functionality."""
        model = SimpleONNModel()
        state_dict = model.state_dict()
        
        # Check state dict contains expected keys
        expected_keys = [
            'layer1.u_matrix', 'layer1.s_matrix', 'layer1.v_matrix',
            'layer2.weight'
        ]
        
        for key in expected_keys:
            assert key in state_dict
    
    def test_model_loading(self):
        """Test model loading from state dict."""
        model1 = SimpleONNModel()
        model2 = SimpleONNModel()
        
        # Save state from model1
        state_dict = model1.state_dict()
        
        # Load into model2
        model2.load_state_dict(state_dict)
        
        # Check parameters are equal
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2)