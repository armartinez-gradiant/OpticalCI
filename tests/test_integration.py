"""
Tests de Integración - PtONN-TESTS

Suite de tests que valida:
- Integración entre diferentes componentes
- Redes fotónicas completas
- End-to-end functionality
- Performance de sistemas completos
- Coherencia física del sistema completo
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, Any, List

# Import de módulos para integración
from torchonn.layers import MZILayer, MZIBlockLinear, MicroringResonator, AddDropMRR
from torchonn.layers import DirectionalCoupler, Photodetector
from torchonn.components import WDMMultiplexer, PhaseChangeCell
from torchonn.models import ONNBaseModel


class SimplePhotonicNN(ONNBaseModel):
    """Red fotónica simple para tests de integración."""
    
    def __init__(self, device=None):
        super().__init__(device=device)
        
        self.input_layer = MZIBlockLinear(
            in_features=4,
            out_features=3,
            mode="usv",
            device=self.device
        )
        
        self.nonlinear_element = MicroringResonator(
            radius=5e-6,
            q_factor=1000,  # Q bajo para estabilidad en tests
            coupling_mode="critical",
            device=self.device
        )
        
        self.output_layer = MZIBlockLinear(
            in_features=3,
            out_features=2,
            mode="weight",
            device=self.device
        )
        
        self.detector = Photodetector(responsivity=0.8, device=self.device)
        
        # Wavelengths para el microring
        self.wavelengths = torch.linspace(1549e-9, 1551e-9, 3, device=self.device, dtype=torch.float32)
    
    def forward(self, x):
        # Procesamiento lineal
        x = self.input_layer(x)
        
        # Elemento no-lineal (microring)
        with torch.no_grad():  # Para estabilidad en tests
            mrr_output = self.nonlinear_element(x, self.wavelengths)
            x = mrr_output['through']
        
        # Procesamiento de salida
        x = self.output_layer(x)
        
        # Detección
        electrical_output = self.detector(x)
        
        return electrical_output


class ComplexPhotonicNN(ONNBaseModel):
    """Red fotónica compleja para tests avanzados."""
    
    def __init__(self, device=None):
        super().__init__(device=device)
        
        # Capa de entrada con MZI unitario
        self.mzi_input = MZILayer(in_features=6, out_features=6, device=self.device)
        
        # Processing con diferentes modos
        self.processing_1 = MZIBlockLinear(6, 4, mode="usv", device=self.device)
        self.processing_2 = MZIBlockLinear(4, 4, mode="phase", device=self.device)
        
        # Sistema WDM
        wdm_wavelengths = [1530e-9, 1540e-9, 1550e-9, 1560e-9]
        self.wdm_system = WDMMultiplexer(wavelengths=wdm_wavelengths, device=self.device)
        
        # Add-Drop para cada canal
        self.add_drops = torch.nn.ModuleList([
            AddDropMRR(radius=5e-6, q_factor=800, device=self.device) for _ in range(4)
        ])
        
        # Coupler para combining
        self.coupler = DirectionalCoupler(splitting_ratio=0.5, device=self.device)
        
        # Detection
        self.detector = Photodetector(responsivity=1.0, device=self.device)
    
    def forward(self, x):
        # Entrada unitaria
        x = self.mzi_input(x)
        
        # Processing
        x = self.processing_1(x)
        x = self.processing_2(x)
        
        # Preparar para WDM (necesitamos 4 canales)
        channels = [x[:, i] for i in range(min(4, x.size(1)))]
        while len(channels) < 4:
            channels.append(torch.zeros_like(channels[0]))
        
        # WDM multiplexing
        multiplexed = self.wdm_system.multiplex(channels)
        
        # Procesamiento con Add-Drop (simplificado para test)
        processed_channels = []
        for i, add_drop in enumerate(self.add_drops):
            channel_signal = multiplexed[:, i:i+1]
            wavelength = torch.tensor([1530e-9 + i*10e-9], device=self.device, dtype=torch.float32)
            add_signal = torch.zeros_like(channel_signal)
            
            with torch.no_grad():
                ad_output = add_drop(channel_signal, add_signal, wavelength)
                processed_channels.append(ad_output['through'][:, 0])
        
        # Combining con coupler (solo primeros 2 canales)
        if len(processed_channels) >= 2:
            combined_1, combined_2 = self.coupler(processed_channels[0], processed_channels[1])
            final_signal = combined_1 + combined_2
        else:
            final_signal = processed_channels[0]
        
        # Detection
        electrical_output = self.detector(final_signal.unsqueeze(1))
        
        return electrical_output


class TestSimpleIntegration:
    """Tests de integración básicos."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def simple_network(self, device):
        """Fixture para red simple."""
        return SimplePhotonicNN(device=device)
    
    def test_simple_network_forward(self, simple_network, device):
        """Test: Forward pass completo funciona."""
        batch_size = 8
        input_data = torch.randn(batch_size, 4, device=device, dtype=torch.float32)
        
        # Forward pass
        output = simple_network(input_data)
        
        # Test dimensiones
        assert output.shape == (batch_size, 2), f"Wrong output shape: {output.shape}"
        
        # Test no NaN/Inf
        assert not torch.any(torch.isnan(output)), "NaN in network output"
        assert not torch.any(torch.isinf(output)), "Inf in network output"
        
        # Test valores razonables (detección produce corrientes positivas)
        assert torch.all(output >= 0), "Negative current from photodetector"
    
    def test_simple_network_gradients(self, simple_network, device):
        """Test: Gradientes fluyen correctamente en la red."""
        batch_size = 4
        input_data = torch.randn(batch_size, 4, device=device, dtype=torch.float32, requires_grad=True)
        
        # Forward + backward
        output = simple_network(input_data)
        loss = torch.mean(output**2)
        loss.backward()
        
        # Test gradientes en input
        assert input_data.grad is not None, "No gradients on input"
        assert not torch.all(input_data.grad == 0), "Input gradients are zero"
        
        # Test gradientes en capas entrenables
        for name, param in simple_network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradients on parameter {name}"
    
    def test_simple_network_different_batch_sizes(self, simple_network, device):
        """Test: Diferentes batch sizes funcionan."""
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 4, device=device, dtype=torch.float32)
            
            output = simple_network(input_data)
            
            assert output.shape == (batch_size, 2), f"Wrong shape for batch_size {batch_size}"
            assert not torch.any(torch.isnan(output)), f"NaN for batch_size {batch_size}"
    
    def test_network_reproducibility(self, device):
        """Test: Red produce resultados reproducibles."""
        # Crear dos redes idénticas
        torch.manual_seed(42)
        net1 = SimplePhotonicNN(device=device)
        
        torch.manual_seed(42)
        net2 = SimplePhotonicNN(device=device)
        
        # Input idéntico
        torch.manual_seed(123)
        input_data = torch.randn(5, 4, device=device, dtype=torch.float32)
        
        # Forward pass
        output1 = net1(input_data)
        output2 = net2(input_data)
        
        # Deben ser idénticos (determinísticos)
        assert torch.allclose(output1, output2, atol=1e-6), "Network not reproducible"


class TestComplexIntegration:
    """Tests de integración compleja."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def complex_network(self, device):
        """Fixture para red compleja."""
        return ComplexPhotonicNN(device=device)
    
    def test_complex_network_forward(self, complex_network, device):
        """Test: Red compleja funciona end-to-end."""
        batch_size = 4
        input_data = torch.randn(batch_size, 6, device=device, dtype=torch.float32)
        
        # Forward pass
        start_time = time.time()
        output = complex_network(input_data)
        forward_time = time.time() - start_time
        
        # Test dimensiones y valores
        assert output.shape[0] == batch_size, "Wrong batch dimension"
        assert not torch.any(torch.isnan(output)), "NaN in complex network"
        assert not torch.any(torch.isinf(output)), "Inf in complex network"
        
        # Test performance razonable
        assert forward_time < 5.0, f"Complex network too slow: {forward_time:.3f}s"
    
    def test_energy_flow_conservation(self, device):
        """Test: Conservación de energía a través de la red."""
        # Red simplificada para test de conservación
        mzi = MZILayer(in_features=4, out_features=4, device=device)
        
        input_data = torch.randn(10, 4, device=device, dtype=torch.float32)
        
        # Energía de entrada
        input_energy = torch.sum(torch.abs(input_data)**2, dim=1)
        
        # Después de MZI unitario
        mzi_output = mzi(input_data)
        mzi_energy = torch.sum(torch.abs(mzi_output)**2, dim=1)
        
        # Test conservación
        energy_ratio = mzi_energy / torch.clamp(input_energy, min=1e-10)
        assert torch.allclose(energy_ratio, torch.ones_like(energy_ratio), atol=1e-3), "Energy not conserved through MZI"
    
    def test_component_interaction(self, device):
        """Test: Interacción correcta entre componentes."""
        # Test MZI -> Microring -> Detector
        mzi = MZIBlockLinear(in_features=3, out_features=3, mode="usv", device=device)
        mrr = MicroringResonator(q_factor=500, device=device)  # Q bajo para estabilidad
        detector = Photodetector(device=device)
        
        # Wavelengths
        wavelengths = torch.linspace(1549e-9, 1551e-9, 3, device=device, dtype=torch.float32)
        
        # Pipeline
        input_data = torch.randn(5, 3, device=device, dtype=torch.float32)
        
        # Stage 1: MZI
        mzi_output = mzi(input_data)
        assert mzi_output.shape == (5, 3)
        
        # Stage 2: Microring
        with torch.no_grad():
            mrr_output = mrr(mzi_output, wavelengths)
        
        through_signal = mrr_output['through']
        assert through_signal.shape == (5, 3)
        assert torch.all(through_signal <= 1.01), "Microring violates energy conservation"
        
        # Stage 3: Detection
        electrical_output = detector(through_signal)
        assert electrical_output.shape == (5, 3)
        assert torch.all(electrical_output >= 0), "Negative electrical signal"


class TestSystemPerformance:
    """Tests de performance del sistema completo."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_large_batch_performance(self, device):
        """Test: Performance con batches grandes."""
        network = SimplePhotonicNN(device=device)
        
        # Batch grande
        large_batch_size = 512
        input_data = torch.randn(large_batch_size, 4, device=device, dtype=torch.float32)
        
        # Warm up
        _ = network(input_data[:32])
        
        # Measure time
        start_time = time.time()
        output = network(input_data)
        forward_time = time.time() - start_time
        
        # Test performance
        assert forward_time < 10.0, f"Large batch too slow: {forward_time:.3f}s"
        assert output.shape == (large_batch_size, 2)
        
        # Test memory efficiency (no memory leaks)
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(device)
            assert memory_used < 1e9, f"Excessive memory usage: {memory_used/1e6:.1f} MB"  # < 1GB
    
    def test_multiple_forward_passes(self, device):
        """Test: Múltiples forward passes consecutivos."""
        network = SimplePhotonicNN(device=device)
        
        # Múltiples forward passes
        n_passes = 50
        batch_size = 16
        
        outputs = []
        for i in range(n_passes):
            input_data = torch.randn(batch_size, 4, device=device, dtype=torch.float32)
            output = network(input_data)
            outputs.append(output)
        
        # Test que todos funcionaron
        assert len(outputs) == n_passes
        
        # Test consistency
        for i, output in enumerate(outputs):
            assert output.shape == (batch_size, 2), f"Wrong shape at pass {i}"
            assert not torch.any(torch.isnan(output)), f"NaN at pass {i}"
    
    def test_component_count_scaling(self, device):
        """Test: Scaling con número de componentes."""
        # Test diferentes configuraciones
        configurations = [
            {"layers": 2, "features": [4, 3, 2]},
            {"layers": 3, "features": [6, 4, 3, 2]},
            {"layers": 4, "features": [8, 6, 4, 3, 2]}
        ]
        
        for config in configurations:
            # Crear red con configuración específica
            class ScalableNetwork(ONNBaseModel):
                def __init__(self, features, device):
                    super().__init__(device=device)
                    self.layers = torch.nn.ModuleList()
                    
                    for i in range(len(features) - 1):
                        layer = MZIBlockLinear(
                            features[i], features[i+1], 
                            mode="weight", device=device
                        )
                        self.layers.append(layer)
                
                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    return x
            
            network = ScalableNetwork(config["features"], device)
            
            # Test funcionamiento
            input_size = config["features"][0]
            output_size = config["features"][-1]
            
            input_data = torch.randn(8, input_size, device=device, dtype=torch.float32)
            output = network(input_data)
            
            assert output.shape == (8, output_size), f"Wrong output shape for {config['layers']} layers"


class TestPhysicsValidation:
    """Tests de validación física del sistema completo."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_end_to_end_energy_budget(self, device):
        """Test: Presupuesto de energía end-to-end."""
        # Red con componentes que conservan energía
        mzi1 = MZILayer(in_features=4, out_features=4, device=device)  # Unitario
        mzi2 = MZILayer(in_features=4, out_features=4, device=device)  # Unitario
        
        input_data = torch.randn(10, 4, device=device, dtype=torch.float32)
        
        # Pipeline
        stage1 = mzi1(input_data)
        stage2 = mzi2(stage1)
        
        # Energías
        input_energy = torch.sum(torch.abs(input_data)**2)
        output_energy = torch.sum(torch.abs(stage2)**2)
        
        energy_ratio = output_energy / input_energy
        assert abs(energy_ratio - 1.0) < 1e-3, f"End-to-end energy not conserved: {energy_ratio:.6f}"
    
    def test_realistic_component_parameters(self, device):
        """Test: Parámetros de componentes son físicamente realistas."""
        # Test microring
        mrr = MicroringResonator(q_factor=2000, device=device)
        assert 5 < mrr.extinction_ratio_theory_db < 30, "ER not realistic"
        
        # Test coupling coefficients
        assert 0.01 < mrr.kappa_critical < 0.5, "Coupling coefficient not realistic"
        
        # Test FSR
        assert 1e-12 < mrr.fsr < 50e-12, "FSR not realistic"  # 1-50 pm range
    
    def test_no_unphysical_outputs(self, device):
        """Test: No hay outputs no físicos en ningún componente."""
        components_to_test = [
            MicroringResonator(device=device),
            AddDropMRR(device=device),
            Photodetector(device=device)
        ]
        
        # Test cada componente
        for component in components_to_test:
            if isinstance(component, (MicroringResonator, AddDropMRR)):
                wavelengths = torch.linspace(1549e-9, 1551e-9, 5, device=device, dtype=torch.float32)
                input_signal = torch.ones(2, 5, device=device, dtype=torch.float32)
                
                if isinstance(component, MicroringResonator):
                    with torch.no_grad():
                        output = component(input_signal, wavelengths)
                    
                    # Test física
                    assert torch.all(output['through'][0] <= 1.01), f"Through > 1 in {type(component).__name__}"
                    assert torch.all(output['drop'][0] <= 1.01), f"Drop > 1 in {type(component).__name__}"
                    assert torch.all(output['through'][0] >= 0), f"Through < 0 in {type(component).__name__}"
                    assert torch.all(output['drop'][0] >= 0), f"Drop < 0 in {type(component).__name__}"
                
                elif isinstance(component, AddDropMRR):
                    add_signal = torch.zeros_like(input_signal)
                    with torch.no_grad():
                        output = component(input_signal, add_signal, wavelengths)
                    
                    assert torch.all(output['through'][0] <= 1.01), "AddDrop through > 1"
                    assert torch.all(output['drop'][0] <= 1.01), "AddDrop drop > 1"
            
            elif isinstance(component, Photodetector):
                optical_input = torch.randn(3, 4, device=device, dtype=torch.float32)
                electrical_output = component(optical_input)
                
                # Test corriente no negativa
                assert torch.all(electrical_output >= 0), "Negative photocurrent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])