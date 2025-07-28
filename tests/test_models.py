"""
Tests para Models - PtONN-TESTS

Suite de tests que valida:
- ONNBaseModel y funcionalidades base
- Funcionalidades de device management
- Reset de parámetros
- Inheritance y extensibilidad
- Integration con PyTorch nn.Module
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

# Import del módulo a testear
from torchonn.models import ONNBaseModel
from torchonn.layers import MZILayer, MZIBlockLinear, MicroringResonator


class SimpleONN(ONNBaseModel):
    """ONN simple para tests."""
    
    def __init__(self, device=None):
        super().__init__(device=device)
        
        self.layer1 = MZIBlockLinear(4, 3, mode="weight", device=self.device)
        self.layer2 = MZIBlockLinear(3, 2, mode="weight", device=self.device)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)  # Activación simple
        x = self.layer2(x)
        return x


class ComplexONN(ONNBaseModel):
    """ONN complejo para tests avanzados."""
    
    def __init__(self, device=None):
        super().__init__(device=device)
        
        # Múltiples tipos de capas
        self.mzi_unitary = MZILayer(6, 6, device=self.device)
        self.processing1 = MZIBlockLinear(6, 4, mode="usv", device=self.device)
        self.processing2 = MZIBlockLinear(4, 3, mode="phase", device=self.device)
        
        # Elemento no-lineal
        self.nonlinear = MicroringResonator(
            q_factor=1000, 
            coupling_mode="critical",
            device=self.device
        )
        
        # Wavelengths para microring
        self.register_buffer(
            'wavelengths', 
            torch.linspace(1549e-9, 1551e-9, 3, device=self.device, dtype=torch.float32)
        )
    
    def forward(self, x):
        # Procesamiento unitario
        x = self.mzi_unitary(x)
        
        # Procesamiento lineal
        x = self.processing1(x)
        x = self.processing2(x)
        
        # Elemento no-lineal
        with torch.no_grad():  # Para estabilidad en tests
            mrr_output = self.nonlinear(x, self.wavelengths)
            x = mrr_output['through']
        
        return x


class TestONNBaseModel:
    """Tests para ONNBaseModel base class."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_base_model_initialization(self, device):
        """Test: Inicialización correcta del modelo base."""
        model = ONNBaseModel(device=device)
        
        # Verificar device assignment
        assert model.device == device
        
        # Verificar herencia de nn.Module
        assert isinstance(model, nn.Module)
        
        # Verificar que está en el device correcto
        for param in model.parameters():
            assert param.device == device
    
    def test_device_auto_detection(self):
        """Test: Detección automática de device."""
        model = ONNBaseModel()  # Sin especificar device
        
        # Device debe ser cuda si está disponible, sino cpu
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert model.device == expected_device
    
    def test_device_string_input(self):
        """Test: Input de device como string."""
        # Test con string
        model_cpu = ONNBaseModel(device="cpu")
        assert model_cpu.device == torch.device("cpu")
        
        if torch.cuda.is_available():
            model_cuda = ONNBaseModel(device="cuda")
            assert model_cuda.device == torch.device("cuda")
    
    def test_reset_parameters_base(self, device):
        """Test: Reset de parámetros en modelo base (sin parámetros)."""
        model = ONNBaseModel(device=device)
        
        # Debe funcionar sin error aunque no tenga parámetros propios
        model.reset_parameters()  # No debe dar error
    
    def test_base_model_is_abstract(self, device):
        """Test: Modelo base no implementa forward (debe ser implementado por subclases)."""
        model = ONNBaseModel(device=device)
        
        # Intentar forward debe dar error (no implementado)
        input_tensor = torch.randn(4, 3, device=device)
        
        with pytest.raises(NotImplementedError):
            model(input_tensor)


class TestSimpleONN:
    """Tests para ONN simple."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def simple_onn(self, device):
        """Fixture para ONN simple."""
        return SimpleONN(device=device)
    
    def test_simple_onn_initialization(self, simple_onn, device):
        """Test: Inicialización correcta de ONN simple."""
        assert simple_onn.device == device
        
        # Verificar capas
        assert hasattr(simple_onn, 'layer1')
        assert hasattr(simple_onn, 'layer2')
        
        # Verificar que las capas están en el device correcto
        assert simple_onn.layer1.device == device
        assert simple_onn.layer2.device == device
    
    def test_simple_onn_forward(self, simple_onn, device):
        """Test: Forward pass de ONN simple."""
        batch_size = 8
        input_tensor = torch.randn(batch_size, 4, device=device, dtype=torch.float32)
        
        # Forward pass
        output = simple_onn(input_tensor)
        
        # Verificar shape
        assert output.shape == (batch_size, 2)
        
        # Verificar no NaN/Inf
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_simple_onn_gradients(self, simple_onn, device):
        """Test: Gradientes funcionan en ONN simple."""
        input_tensor = torch.randn(4, 4, device=device, dtype=torch.float32, requires_grad=True)
        
        # Forward + backward
        output = simple_onn(input_tensor)
        loss = torch.mean(output**2)
        loss.backward()
        
        # Verificar gradientes en input
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
        
        # Verificar gradientes en parámetros del modelo
        for name, param in simple_onn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradients in parameter: {name}"
    
    def test_simple_onn_parameter_reset(self, simple_onn):
        """Test: Reset de parámetros funciona."""
        # Obtener parámetros iniciales
        initial_params = {}
        for name, param in simple_onn.named_parameters():
            initial_params[name] = param.clone()
        
        # Modificar parámetros
        with torch.no_grad():
            for param in simple_onn.parameters():
                param.fill_(1.0)
        
        # Reset
        simple_onn.reset_parameters()
        
        # Verificar que los parámetros cambiaron
        for name, param in simple_onn.named_parameters():
            assert not torch.allclose(param, initial_params[name]), f"Parameter {name} not reset"
    
    def test_simple_onn_training_mode(self, simple_onn, device):
        """Test: Modos training/eval funcionan."""
        input_tensor = torch.randn(3, 4, device=device, dtype=torch.float32)
        
        # Training mode
        simple_onn.train()
        assert simple_onn.training
        output_train = simple_onn(input_tensor)
        
        # Eval mode
        simple_onn.eval()
        assert not simple_onn.training
        output_eval = simple_onn(input_tensor)
        
        # Outputs pueden ser diferentes dependiendo de las capas
        # Pero ambos deben tener shape correcto
        assert output_train.shape == output_eval.shape


class TestComplexONN:
    """Tests para ONN complejo."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def complex_onn(self, device):
        """Fixture para ONN complejo."""
        return ComplexONN(device=device)
    
    def test_complex_onn_initialization(self, complex_onn, device):
        """Test: Inicialización de ONN complejo."""
        assert complex_onn.device == device
        
        # Verificar todas las capas
        expected_layers = ['mzi_unitary', 'processing1', 'processing2', 'nonlinear']
        for layer_name in expected_layers:
            assert hasattr(complex_onn, layer_name), f"Missing layer: {layer_name}"
        
        # Verificar buffer de wavelengths
        assert hasattr(complex_onn, 'wavelengths')
        assert complex_onn.wavelengths.device == device
    
    def test_complex_onn_forward(self, complex_onn, device):
        """Test: Forward pass de ONN complejo."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 6, device=device, dtype=torch.float32)
        
        # Forward pass
        output = complex_onn(input_tensor)
        
        # Verificar shape (6 -> 4 -> 3, luego microring mantiene 3)
        assert output.shape == (batch_size, 3)
        
        # Verificar valores físicos (microring debe dar [0, 1])
        assert torch.all(output <= 1.01), "Output above physical range"
        assert torch.all(output >= 0), "Output below physical range"
    
    def test_complex_onn_component_interaction(self, complex_onn, device):
        """Test: Interacción entre componentes."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 6, device=device, dtype=torch.float32)
        
        # Forward pass paso a paso
        x = input_tensor
        
        # Paso 1: MZI unitario (debe conservar energía)
        x_mzi = complex_onn.mzi_unitary(x)
        
        input_energy = torch.sum(torch.abs(input_tensor)**2, dim=1)
        mzi_energy = torch.sum(torch.abs(x_mzi)**2, dim=1)
        energy_ratio = mzi_energy / torch.clamp(input_energy, min=1e-10)
        
        assert torch.allclose(energy_ratio, torch.ones_like(energy_ratio), atol=1e-3), "MZI not unitary"
        
        # Resto del procesamiento
        x = complex_onn.processing1(x_mzi)
        x = complex_onn.processing2(x)
        
        # Microring (con validación física)
        with torch.no_grad():
            mrr_output = complex_onn.nonlinear(x, complex_onn.wavelengths)
            final_output = mrr_output['through']
        
        # Verificar rango físico del microring
        assert torch.all(final_output <= 1.01), "Microring output > 1"
        assert torch.all(final_output >= 0), "Microring output < 0"
    
    def test_complex_onn_device_consistency(self, complex_onn, device):
        """Test: Consistencia de device en todos los componentes."""
        # Verificar que todos los parámetros están en el device correcto
        for name, param in complex_onn.named_parameters():
            assert param.device == device, f"Parameter {name} on wrong device: {param.device}"
        
        # Verificar que todos los buffers están en el device correcto
        for name, buffer in complex_onn.named_buffers():
            assert buffer.device == device, f"Buffer {name} on wrong device: {buffer.device}"


class TestONNInheritance:
    """Tests para herencia y extensibilidad."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_custom_onn_inheritance(self, device):
        """Test: Herencia personalizada funciona."""
        
        class CustomONN(ONNBaseModel):
            def __init__(self, custom_param=42, device=None):
                super().__init__(device=device)
                self.custom_param = custom_param
                self.layer = MZIBlockLinear(3, 2, mode="weight", device=self.device)
            
            def forward(self, x):
                return self.layer(x) * self.custom_param
            
            def get_custom_info(self):
                return f"Custom ONN with param: {self.custom_param}"
        
        # Crear instancia
        custom_onn = CustomONN(custom_param=100, device=device)
        
        # Verificar herencia
        assert isinstance(custom_onn, ONNBaseModel)
        assert isinstance(custom_onn, nn.Module)
        
        # Verificar funcionalidad personalizada
        assert custom_onn.custom_param == 100
        assert custom_onn.device == device
        assert "Custom ONN with param: 100" in custom_onn.get_custom_info()
        
        # Verificar forward personalizado
        input_tensor = torch.randn(2, 3, device=device, dtype=torch.float32)
        output = custom_onn(input_tensor)
        
        assert output.shape == (2, 2)
        # Output debe estar escalado por custom_param
        assert torch.max(torch.abs(output)) > 10  # Escalado por ~100
    
    def test_multiple_inheritance_compatibility(self, device):
        """Test: Compatibilidad con herencia múltiple (si es necesaria)."""
        
        class MixinClass:
            def mixin_method(self):
                return "mixin_called"
        
        class MixedONN(ONNBaseModel, MixinClass):
            def __init__(self, device=None):
                ONNBaseModel.__init__(self, device=device)
                self.layer = MZIBlockLinear(2, 1, mode="weight", device=self.device)
            
            def forward(self, x):
                return self.layer(x)
        
        # Crear instancia
        mixed_onn = MixedONN(device=device)
        
        # Verificar herencia múltiple
        assert isinstance(mixed_onn, ONNBaseModel)
        assert isinstance(mixed_onn, MixinClass)
        assert mixed_onn.mixin_method() == "mixin_called"
        
        # Verificar funcionalidad ONN
        input_tensor = torch.randn(1, 2, device=device, dtype=torch.float32)
        output = mixed_onn(input_tensor)
        assert output.shape == (1, 1)


class TestONNSerialization:
    """Tests para serialización y guardado/cargado de modelos."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_state_dict_save_load(self, device):
        """Test: Guardar y cargar state_dict."""
        # Crear modelo original
        original_model = SimpleONN(device=device)
        input_tensor = torch.randn(2, 4, device=device, dtype=torch.float32)
        
        # Forward pass original
        original_output = original_model(input_tensor)
        
        # Guardar state dict
        state_dict = original_model.state_dict()
        
        # Crear nuevo modelo y cargar state dict
        new_model = SimpleONN(device=device)
        new_model.load_state_dict(state_dict)
        
        # Forward pass nuevo modelo
        new_output = new_model(input_tensor)
        
        # Outputs deben ser idénticos
        assert torch.allclose(original_output, new_output, atol=1e-6), "State dict load/save failed"
    
    def test_parameter_persistence(self, device):
        """Test: Persistencia de parámetros personalizados."""
        
        class ModelWithCustomParams(ONNBaseModel):
            def __init__(self, device=None):
                super().__init__(device=device)
                self.layer = MZIBlockLinear(3, 2, mode="weight", device=self.device)
                
                # Parámetro personalizado
                self.register_parameter(
                    'custom_weight', 
                    nn.Parameter(torch.randn(2, 3, device=self.device))
                )
                
                # Buffer personalizado
                self.register_buffer(
                    'custom_buffer',
                    torch.ones(2, device=self.device)
                )
            
            def forward(self, x):
                x = self.layer(x)
                x = torch.mm(x, self.custom_weight.t())
                x = x + self.custom_buffer.unsqueeze(0)
                return x
        
        # Crear y usar modelo
        model = ModelWithCustomParams(device=device)
        input_tensor = torch.randn(1, 3, device=device, dtype=torch.float32)
        original_output = model(input_tensor)
        
        # Verificar que custom parameters están en state_dict
        state_dict = model.state_dict()
        assert 'custom_weight' in state_dict
        assert 'custom_buffer' in state_dict
        
        # Cargar en nuevo modelo
        new_model = ModelWithCustomParams(device=device)
        new_model.load_state_dict(state_dict)
        new_output = new_model(input_tensor)
        
        # Verificar consistencia
        assert torch.allclose(original_output, new_output, atol=1e-6)


class TestONNCompatibility:
    """Tests de compatibilidad con PyTorch estándar."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_pytorch_optimizer_compatibility(self, device):
        """Test: Compatibilidad con optimizadores de PyTorch."""
        model = SimpleONN(device=device)
        
        # Test con diferentes optimizadores
        optimizers = [
            torch.optim.SGD(model.parameters(), lr=0.01),
            torch.optim.Adam(model.parameters(), lr=0.01),
            torch.optim.RMSprop(model.parameters(), lr=0.01)
        ]
        
        for optimizer in optimizers:
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward + backward
            input_tensor = torch.randn(2, 4, device=device, dtype=torch.float32)
            output = model(input_tensor)
            loss = torch.mean(output**2)
            loss.backward()
            
            # Verificar que hay gradientes
            has_gradients = any(p.grad is not None for p in model.parameters())
            assert has_gradients, f"No gradients with {type(optimizer).__name__}"
            
            # Optimizer step
            optimizer.step()
    
    def test_pytorch_loss_compatibility(self, device):
        """Test: Compatibilidad con funciones de pérdida de PyTorch."""
        model = SimpleONN(device=device)
        input_tensor = torch.randn(4, 4, device=device, dtype=torch.float32)
        output = model(input_tensor)
        
        # Target para loss functions
        target = torch.randn(4, 2, device=device, dtype=torch.float32)
        
        # Test diferentes loss functions
        loss_functions = [
            nn.MSELoss(),
            nn.L1Loss(),
            nn.SmoothL1Loss()
        ]
        
        for loss_fn in loss_functions:
            loss = loss_fn(output, target)
            
            # Verificar que loss es escalar
            assert loss.dim() == 0, f"Loss not scalar with {type(loss_fn).__name__}"
            
            # Verificar que loss es finito
            assert torch.isfinite(loss), f"Loss not finite with {type(loss_fn).__name__}"
    
    def test_pytorch_dataloader_compatibility(self, device):
        """Test: Compatibilidad con DataLoader de PyTorch."""
        from torch.utils.data import DataLoader, TensorDataset
        
        model = SimpleONN(device=device)
        
        # Crear dataset simple
        data = torch.randn(20, 4, dtype=torch.float32)
        targets = torch.randn(20, 2, dtype=torch.float32)
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test training loop simple
        model.train()
        for batch_data, batch_targets in dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            
            output = model(batch_data)
            
            # Verificar shapes
            assert output.shape == batch_targets.shape
            
            # Test que se puede calcular loss
            loss = nn.MSELoss()(output, batch_targets)
            assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])