"""
Tests para Microring Resonators - PtONN-TESTS

Suite completa de tests que valida:
- Conservación de energía (drop ≤ 1.0)
- Extinction ratios realistas
- Parámetros físicamente coherentes
- Validación automática
- Edge cases y robustez
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

# Import del módulo a testear
from torchonn.layers import MicroringResonator, AddDropMRR


class TestMicroringResonator:
    """Tests para MicroringResonator con validación física."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def standard_mrr(self, device):
        """Fixture para microring estándar."""
        return MicroringResonator(
            radius=10e-6,
            q_factor=5000,
            center_wavelength=1550e-9,
            coupling_mode="critical",
            device=device
        )
    
    @pytest.fixture
    def wavelengths_short(self, device):
        """Fixture para wavelengths cortos (testing rápido)."""
        return torch.linspace(1549e-9, 1551e-9, 100, device=device, dtype=torch.float32)
    
    @pytest.fixture
    def input_signal_short(self, device):
        """Fixture para señal de entrada corta."""
        return torch.ones(1, 100, device=device, dtype=torch.float32)
    
    def test_microring_initialization(self, device):
        """Test: Inicialización correcta del microring."""
        mrr = MicroringResonator(
            radius=10e-6,
            q_factor=1000,
            coupling_mode="critical",
            device=device
        )
        
        # Verificar parámetros básicos
        assert mrr.radius == 10e-6
        assert mrr.q_factor == 1000
        assert mrr.center_wavelength == 1550e-9
        assert mrr.device == device
        
        # Verificar parámetros calculados
        expected_alpha = np.exp(-np.pi / 1000)
        assert abs(mrr.alpha - expected_alpha) < 1e-6
        
        expected_kappa_critical = np.sqrt(1 - expected_alpha**2)
        assert abs(mrr.kappa_critical - expected_kappa_critical) < 1e-6
        
        # Verificar que extinction ratio teórico es realista
        assert 5 < mrr.extinction_ratio_theory_db < 25
    
    def test_energy_conservation(self, standard_mrr, wavelengths_short, input_signal_short):
        """Test: Conservación de energía estricta."""
        with torch.no_grad():
            output = standard_mrr(input_signal_short, wavelengths_short)
            
        through_response = output['through'][0]
        drop_response = output['drop'][0]
        
        # Test 1: Ningún puerto puede exceder 1.0
        assert torch.all(through_response <= 1.01), f"Through > 1.0: max = {torch.max(through_response):.3f}"
        assert torch.all(drop_response <= 1.01), f"Drop > 1.0: max = {torch.max(drop_response):.3f}"
        
        # Test 2: Valores no negativos
        assert torch.all(through_response >= 0), "Through < 0 detectado"
        assert torch.all(drop_response >= 0), "Drop < 0 detectado"
        
        # Test 3: Conservación de energía total
        total_energy = through_response + drop_response
        max_energy = torch.max(total_energy)
        mean_energy = torch.mean(total_energy)
        
        assert max_energy <= 1.01, f"Conservación violada: max_energy = {max_energy:.3f}"
        assert mean_energy > 0.8, f"Pérdidas excesivas: mean_energy = {mean_energy:.3f}"
    
    def test_extinction_ratio_realistic(self, standard_mrr, device):
        """Test: Extinction ratio en rango realista."""
        # Usar wavelengths recomendados
        wavelengths = standard_mrr.get_recommended_wavelengths(200)
        input_signal = torch.ones(1, 200, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output = standard_mrr(input_signal, wavelengths)
            
        through_response = output['through'][0]
        
        # Calcular extinction ratio
        min_through = torch.min(through_response)
        max_through = torch.max(through_response)
        
        if min_through > 1e-10:
            er_measured = max_through / min_through
            er_db = 10 * torch.log10(er_measured)
            
            # Test: ER en rango físicamente realista
            assert 5 < er_db < 35, f"ER fuera de rango: {er_db:.1f} dB"
            
            # Test: ER coherente con teoría (tolerancia realista)
            er_theory = standard_mrr.extinction_ratio_theory_db
            er_error = abs(er_db - er_theory)
            tolerance = 6 + max(3, standard_mrr.q_factor/1000)  # Tolerancia adaptativa
            
            assert er_error < tolerance, f"ER incoherente: {er_db:.1f} vs {er_theory:.1f} dB (error: {er_error:.1f} > {tolerance:.1f})"
    
    def test_physics_validation_automatic(self, standard_mrr):
        """Test: Validación física automática pasa."""
        validation = standard_mrr.validate_physics()
        
        # Test: Conservación de energía
        assert validation['energy_conserved'], f"Energy conservation failed: {validation['energy_conservation']:.3f}"
        
        # Test: Extinction ratio coherente
        assert validation['extinction_ratio_coherent'], (
            f"ER coherence failed: {validation['extinction_ratio_measured_db']:.1f} vs "
            f"{validation['extinction_ratio_theory_db']:.1f} dB"
        )
        
        # Test: Resonancia centrada
        assert validation['resonance_centered'], f"Resonance not centered: {validation['resonance_wavelength_nm']:.3f} nm"
    
    def test_different_q_factors(self, device):
        """Test: Comportamiento con diferentes Q factors."""
        q_factors = [500, 1000, 2000, 5000]
        
        for q in q_factors:
            mrr = MicroringResonator(q_factor=q, coupling_mode="critical", device=device)
            
            # Test wavelengths cortos para velocidad
            wavelengths = torch.linspace(1549e-9, 1551e-9, 50, device=device, dtype=torch.float32)
            input_signal = torch.ones(1, 50, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                output = mrr(input_signal, wavelengths)
            
            through_response = output['through'][0]
            drop_response = output['drop'][0]
            
            # Verificar conservación para todos los Q
            assert torch.all(through_response <= 1.01), f"Q={q}: Through > 1.0"
            assert torch.all(drop_response <= 1.01), f"Q={q}: Drop > 1.0"
            
            # Verificar ER teórico escala con Q
            assert mrr.extinction_ratio_theory_db > 5, f"Q={q}: ER muy bajo"
            assert mrr.extinction_ratio_theory_db < 30, f"Q={q}: ER muy alto"
    
    def test_coupling_modes(self, device):
        """Test: Diferentes modos de coupling."""
        modes = ["critical", "under", "over"]
        
        for mode in modes:
            mrr = MicroringResonator(coupling_mode=mode, device=device)
            
            # Test básico de funcionamiento
            wavelengths = torch.linspace(1549e-9, 1551e-9, 50, device=device, dtype=torch.float32)
            input_signal = torch.ones(1, 50, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                output = mrr(input_signal, wavelengths)
            
            # Verificar conservación de energía siempre
            through_response = output['through'][0]
            drop_response = output['drop'][0]
            
            assert torch.all(through_response <= 1.01), f"Mode {mode}: Through > 1.0"
            assert torch.all(drop_response <= 1.01), f"Mode {mode}: Drop > 1.0"
    
    def test_wavelength_ranges(self, standard_mrr, device):
        """Test: Diferentes rangos de wavelength."""
        # Test 1: Rango muy estrecho
        wl_narrow = torch.linspace(1549.9e-9, 1550.1e-9, 50, device=device, dtype=torch.float32)
        input_narrow = torch.ones(1, 50, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output_narrow = standard_mrr(input_narrow, wl_narrow)
        
        assert torch.all(output_narrow['through'][0] <= 1.01)
        assert torch.all(output_narrow['drop'][0] <= 1.01)
        
        # Test 2: Rango amplio
        wl_wide = torch.linspace(1540e-9, 1560e-9, 100, device=device, dtype=torch.float32)
        input_wide = torch.ones(1, 100, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output_wide = standard_mrr(input_wide, wl_wide)
        
        assert torch.all(output_wide['through'][0] <= 1.01)
        assert torch.all(output_wide['drop'][0] <= 1.01)
    
    def test_batch_processing(self, standard_mrr, device):
        """Test: Procesamiento en batch."""
        wavelengths = torch.linspace(1549e-9, 1551e-9, 50, device=device, dtype=torch.float32)
        
        # Test diferentes batch sizes
        for batch_size in [1, 4, 16]:
            input_signal = torch.ones(batch_size, 50, device=device, dtype=torch.float32)
            
            with torch.no_grad():
                output = standard_mrr(input_signal, wavelengths)
            
            # Verificar shapes
            assert output['through'].shape == (batch_size, 50)
            assert output['drop'].shape == (batch_size, 50)
            
            # Verificar conservación para todos los batches
            for b in range(batch_size):
                through_b = output['through'][b]
                drop_b = output['drop'][b]
                
                assert torch.all(through_b <= 1.01), f"Batch {b}: Through > 1.0"
                assert torch.all(drop_b <= 1.01), f"Batch {b}: Drop > 1.0"
    
    def test_edge_cases(self, device):
        """Test: Edge cases y robustez."""
        # Test 1: Q muy bajo
        mrr_low_q = MicroringResonator(q_factor=100, device=device)
        wavelengths = torch.linspace(1549e-9, 1551e-9, 50, device=device, dtype=torch.float32)
        input_signal = torch.ones(1, 50, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output = mrr_low_q(input_signal, wavelengths)
        
        assert torch.all(output['through'][0] <= 1.01)
        assert torch.all(output['drop'][0] <= 1.01)
        
        # Test 2: Wavelength única
        wl_single = torch.tensor([1550e-9], device=device, dtype=torch.float32)
        input_single = torch.ones(1, 1, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output_single = standard_mrr(input_single, wl_single)
        
        assert output_single['through'].shape == (1, 1)
        assert output_single['drop'].shape == (1, 1)
    
    def test_no_nan_inf(self, standard_mrr, wavelengths_short, input_signal_short):
        """Test: No hay NaN o Inf en outputs."""
        with torch.no_grad():
            output = standard_mrr(input_signal_short, wavelengths_short)
        
        through_response = output['through'][0]
        drop_response = output['drop'][0]
        
        # Test NaN
        assert not torch.any(torch.isnan(through_response)), "NaN detectado en through"
        assert not torch.any(torch.isnan(drop_response)), "NaN detectado en drop"
        
        # Test Inf
        assert not torch.any(torch.isinf(through_response)), "Inf detectado en through"
        assert not torch.any(torch.isinf(drop_response)), "Inf detectado en drop"


class TestAddDropMRR:
    """Tests para Add-Drop Microring Resonator."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def add_drop_mrr(self, device):
        """Fixture para Add-Drop MRR estándar."""
        return AddDropMRR(
            radius=8e-6,
            q_factor=3000,
            center_wavelength=1550e-9,
            device=device
        )
    
    def test_add_drop_initialization(self, device):
        """Test: Inicialización Add-Drop MRR."""
        add_drop = AddDropMRR(radius=8e-6, q_factor=2000, device=device)
        
        assert add_drop.radius == 8e-6
        assert add_drop.q_factor == 2000
        assert add_drop.center_wavelength == 1550e-9
        assert add_drop.device == device
    
    def test_add_drop_energy_conservation(self, add_drop_mrr, device):
        """Test: Conservación de energía en Add-Drop."""
        wavelengths = torch.tensor([1549e-9, 1550e-9, 1551e-9], device=device, dtype=torch.float32)
        input_signal = torch.ones(1, 3, device=device, dtype=torch.float32)
        add_signal = torch.zeros(1, 3, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output = add_drop_mrr(input_signal, add_signal, wavelengths)
        
        through_out = output['through'][0]
        drop_out = output['drop'][0]
        
        # Verificar rango físico
        assert torch.all(through_out <= 1.01), f"Through > 1.0: {torch.max(through_out):.3f}"
        assert torch.all(drop_out <= 1.01), f"Drop > 1.0: {torch.max(drop_out):.3f}"
        assert torch.all(through_out >= 0), "Through < 0"
        assert torch.all(drop_out >= 0), "Drop < 0"
    
    def test_add_drop_resonance_behavior(self, add_drop_mrr, device):
        """Test: Comportamiento en resonancia vs off-resonance."""
        # Wavelengths: off-resonance, resonance, off-resonance
        wavelengths = torch.tensor([1548e-9, 1550e-9, 1552e-9], device=device, dtype=torch.float32)
        input_signal = torch.ones(1, 3, device=device, dtype=torch.float32)
        add_signal = torch.zeros(1, 3, device=device, dtype=torch.float32)
        
        with torch.no_grad():
            output = add_drop_mrr(input_signal, add_signal, wavelengths)
        
        through_out = output['through'][0]
        drop_out = output['drop'][0]
        
        # En resonancia (índice 1): through debe ser menor, drop mayor
        resonance_idx = 1
        off_resonance_idx = 0
        
        assert through_out[resonance_idx] < through_out[off_resonance_idx], "Through no disminuye en resonancia"
        assert drop_out[resonance_idx] > drop_out[off_resonance_idx], "Drop no aumenta en resonancia"


# Tests de performance básicos
class TestMicroringPerformance:
    """Tests de performance para Microring."""
    
    def test_forward_pass_speed(self, device):
        """Test: Velocidad de forward pass aceptable."""
        import time
        
        mrr = MicroringResonator(device=device)
        wavelengths = torch.linspace(1549e-9, 1551e-9, 1000, device=device, dtype=torch.float32)
        input_signal = torch.ones(10, 1000, device=device, dtype=torch.float32)
        
        # Warm up
        with torch.no_grad():
            _ = mrr(input_signal, wavelengths)
        
        # Measure time
        start_time = time.time()
        with torch.no_grad():
            _ = mrr(input_signal, wavelengths)
        forward_time = time.time() - start_time
        
        # Test: Forward pass debe ser < 1 segundo para 1000 wavelengths
        assert forward_time < 1.0, f"Forward pass muy lento: {forward_time:.3f}s"
    
    def test_validation_speed(self, device):
        """Test: Velocidad de validación aceptable."""
        import time
        
        mrr = MicroringResonator(device=device)
        
        start_time = time.time()
        validation = mrr.validate_physics()
        validation_time = time.time() - start_time
        
        # Test: Validación debe ser < 0.5 segundos
        assert validation_time < 0.5, f"Validación muy lenta: {validation_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])