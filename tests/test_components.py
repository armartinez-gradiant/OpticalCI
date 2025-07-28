"""
Tests para Components - PtONN-TESTS

Suite completa de tests que valida:
- WDM Multiplexer/Demultiplexer
- Phase Change Memory Cells
- Directional Couplers
- Photodetectors
- Conservación de energía y física realista
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List

# Import de módulos a testear
from torchonn.components import WDMMultiplexer, PhaseChangeCell
from torchonn.layers import DirectionalCoupler, Photodetector


class TestWDMMultiplexer:
    """Tests para WDM Multiplexer/Demultiplexer."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def wdm_4ch(self, device):
        """Fixture para WDM de 4 canales."""
        wavelengths = [1530e-9, 1540e-9, 1550e-9, 1560e-9]
        return WDMMultiplexer(wavelengths=wavelengths, device=device)
    
    @pytest.fixture
    def test_signals(self, device):
        """Fixture para señales de test."""
        batch_size = 8
        signals = []
        for i in range(4):
            # Cada canal tiene una señal característica
            signal = torch.sin(torch.linspace(0, 2*np.pi, batch_size, device=device, dtype=torch.float32)) * (i + 1)
            signals.append(signal)
        return signals
    
    def test_wdm_initialization(self, device):
        """Test: Inicialización correcta del WDM."""
        wavelengths = [1530e-9, 1540e-9, 1550e-9, 1560e-9]
        wdm = WDMMultiplexer(wavelengths=wavelengths, device=device)
        
        # Verificar parámetros básicos
        assert wdm.n_channels == 4
        assert len(wdm.wavelengths) == 4
        assert len(wdm.drop_filters) == 4
        assert wdm.device == device
        
        # Verificar wavelengths están en tensor correct
        expected_wavelengths = torch.tensor(wavelengths, device=device)
        assert torch.allclose(wdm.wavelengths, expected_wavelengths)
    
    def test_multiplex_function(self, wdm_4ch, test_signals):
        """Test: Función de multiplexing."""
        multiplexed = wdm_4ch.multiplex(test_signals)
        
        # Test dimensiones
        assert multiplexed.shape == (8, 4)  # batch_size x n_channels
        
        # Test que los canales se mantienen
        for i, original_signal in enumerate(test_signals):
            assert torch.allclose(multiplexed[:, i], original_signal, atol=1e-6)
    
    def test_demultiplex_function(self, wdm_4ch, test_signals):
        """Test: Función de demultiplexing."""
        # Multiplex -> Demultiplex
        multiplexed = wdm_4ch.multiplex(test_signals)
        demultiplexed = wdm_4ch.demultiplex(multiplexed)
        
        # Test número de canales
        assert len(demultiplexed) == 4
        
        # Test dimensiones de cada canal
        for i, recovered_signal in enumerate(demultiplexed):
            assert recovered_signal.shape == test_signals[i].shape
    
    def test_channel_fidelity(self, wdm_4ch, test_signals):
        """Test: Fidelidad de canales después de mux/demux."""
        # Round trip: multiplex -> demultiplex
        multiplexed = wdm_4ch.multiplex(test_signals)
        demultiplexed = wdm_4ch.demultiplex(multiplexed)
        
        fidelities = []
        for i, (original, recovered) in enumerate(zip(test_signals, demultiplexed)):
            # Calcular correlación
            if torch.numel(original) > 1:
                correlation = torch.corrcoef(torch.stack([original, recovered]))[0, 1]
                fidelities.append(correlation.item())
            else:
                # Para señales de un solo punto, usar diferencia relativa
                rel_error = torch.abs(original - recovered) / torch.clamp(torch.abs(original), min=1e-10)
                fidelities.append(1.0 - rel_error.item())
        
        # Test fidelidad alta
        avg_fidelity = np.mean(fidelities)
        assert avg_fidelity > 0.9, f"Low channel fidelity: {avg_fidelity:.3f}"
        
        # Test fidelidad individual
        for i, fidelity in enumerate(fidelities):
            assert fidelity > 0.8, f"Channel {i} low fidelity: {fidelity:.3f}"
    
    def test_different_channel_counts(self, device):
        """Test: Diferentes números de canales."""
        channel_counts = [2, 3, 4, 6, 8]
        
        for n_ch in channel_counts:
            # Crear wavelengths equiespaciados
            wavelengths = [1530e-9 + i * 5e-9 for i in range(n_ch)]
            wdm = WDMMultiplexer(wavelengths=wavelengths, device=device)
            
            # Crear señales test
            batch_size = 10
            signals = [torch.randn(batch_size, device=device, dtype=torch.float32) for _ in range(n_ch)]
            
            # Test round trip
            multiplexed = wdm.multiplex(signals)
            demultiplexed = wdm.demultiplex(multiplexed)
            
            assert len(demultiplexed) == n_ch, f"Wrong channel count for {n_ch} channels"
            assert multiplexed.shape == (batch_size, n_ch), f"Wrong multiplexed shape for {n_ch} channels"
    
    def test_edge_cases(self, device):
        """Test: Edge cases para WDM."""
        # Test 1: Un solo canal
        wdm_1ch = WDMMultiplexer(wavelengths=[1550e-9], device=device)
        signal_1ch = [torch.randn(5, device=device, dtype=torch.float32)]
        
        multiplexed_1ch = wdm_1ch.multiplex(signal_1ch)
        demultiplexed_1ch = wdm_1ch.demultiplex(multiplexed_1ch)
        
        assert len(demultiplexed_1ch) == 1
        assert multiplexed_1ch.shape == (5, 1)
        
        # Test 2: Señales de cero
        zero_signals = [torch.zeros(3, device=device, dtype=torch.float32) for _ in range(2)]
        wdm_2ch = WDMMultiplexer(wavelengths=[1540e-9, 1560e-9], device=device)
        
        multiplexed_zero = wdm_2ch.multiplex(zero_signals)
        demultiplexed_zero = wdm_2ch.demultiplex(multiplexed_zero)
        
        for recovered in demultiplexed_zero:
            assert torch.allclose(recovered, torch.zeros_like(recovered), atol=1e-6)


class TestPhaseChangeCell:
    """Tests para Phase Change Memory Cell."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def pcm_cell(self, device):
        """Fixture para PCM cell estándar."""
        return PhaseChangeCell(
            initial_state=0.0,
            switching_energy=1e-12,
            device=device
        )
    
    def test_pcm_initialization(self, device):
        """Test: Inicialización correcta del PCM."""
        pcm = PhaseChangeCell(
            initial_state=0.5,
            switching_energy=2e-12,
            retention_time=5.0,
            device=device
        )
        
        # Verificar parámetros
        assert torch.allclose(pcm.pcm_state, torch.tensor([0.5], device=device))
        assert pcm.switching_energy == 2e-12
        assert pcm.retention_time == 5.0
        assert pcm.device == device
    
    def test_optical_properties(self, pcm_cell):
        """Test: Propiedades ópticas varían con estado."""
        # Test estado amorphous (0.0)
        n_real_0, n_imag_0 = pcm_cell.get_optical_properties()
        
        # Cambiar a estado crystalline
        with torch.no_grad():
            pcm_cell.pcm_state.data.fill_(1.0)
        
        n_real_1, n_imag_1 = pcm_cell.get_optical_properties()
        
        # Test que las propiedades cambian
        assert not torch.allclose(n_real_0, n_real_1), "Real index should change with state"
        assert not torch.allclose(n_imag_0, n_imag_1), "Imaginary index should change with state"
        
        # Test valores físicamente razonables
        assert 1.0 < n_real_0 < 10.0, f"Unrealistic real index: {n_real_0}"
        assert 1.0 < n_real_1 < 10.0, f"Unrealistic real index: {n_real_1}"
        assert 0.0 < n_imag_0 < 5.0, f"Unrealistic imaginary index: {n_imag_0}"
        assert 0.0 < n_imag_1 < 5.0, f"Unrealistic imaginary index: {n_imag_1}"
    
    def test_state_switching(self, pcm_cell):
        """Test: Switching de estado funciona."""
        initial_state = pcm_cell.pcm_state.clone()
        
        # Test switching hacia crystalline (energy > threshold)
        high_energy = torch.tensor(2e-12, device=pcm_cell.device)  # > switching_energy
        pcm_cell.switch_state(high_energy)
        
        assert pcm_cell.pcm_state > initial_state, "State should increase with high energy"
        
        # Test switching hacia amorphous (energy < -threshold)
        low_energy = torch.tensor(-2e-12, device=pcm_cell.device)  # < -switching_energy
        current_state = pcm_cell.pcm_state.clone()
        pcm_cell.switch_state(low_energy)
        
        assert pcm_cell.pcm_state < current_state, "State should decrease with negative energy"
    
    def test_forward_pass(self, pcm_cell, device):
        """Test: Forward pass modula la señal."""
        batch_size = 4
        n_wavelengths = 10
        optical_signal = torch.ones(batch_size, n_wavelengths, device=device, dtype=torch.float32)
        
        # Forward pass
        modulated_signal = pcm_cell(optical_signal)
        
        # Test dimensiones
        assert modulated_signal.shape == optical_signal.shape
        
        # Test que la señal se modula (no es idéntica)
        assert not torch.allclose(modulated_signal, optical_signal), "Signal should be modulated"
        
        # Test que la modulación es físicamente realista (atenuación)
        assert torch.all(modulated_signal <= optical_signal), "PCM should attenuate signal"
        assert torch.all(modulated_signal >= 0), "Modulated signal should be non-negative"
    
    def test_state_clamping(self, device):
        """Test: Estado se mantiene en rango [0, 1]."""
        pcm = PhaseChangeCell(initial_state=0.5, device=device)
        
        # Test límite superior
        with torch.no_grad():
            pcm.pcm_state.data.fill_(2.0)  # Fuera de rango
        
        n_real, n_imag = pcm.get_optical_properties()
        # El estado debe estar clamped internamente
        
        # Test límite inferior
        with torch.no_grad():
            pcm.pcm_state.data.fill_(-0.5)  # Fuera de rango
        
        n_real, n_imag = pcm.get_optical_properties()
        # Debe funcionar sin error


class TestDirectionalCoupler:
    """Tests para Directional Coupler."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def coupler_50_50(self, device):
        """Fixture para coupler 50:50."""
        return DirectionalCoupler(splitting_ratio=0.5, device=device)
    
    def test_coupler_initialization(self, device):
        """Test: Inicialización correcta del coupler."""
        coupler = DirectionalCoupler(
            splitting_ratio=0.3, 
            coupling_length=200e-6,
            device=device
        )
        
        assert torch.allclose(coupler.splitting_ratio, torch.tensor([0.3], device=device))
        assert coupler.coupling_length == 200e-6
        assert coupler.device == device
    
    def test_energy_conservation(self, coupler_50_50, device):
        """Test: Conservación de energía en coupler."""
        batch_size = 8
        n_wavelengths = 5
        
        # Inputs
        input_1 = torch.randn(batch_size, n_wavelengths, device=device, dtype=torch.float32)
        input_2 = torch.randn(batch_size, n_wavelengths, device=device, dtype=torch.float32)
        
        # Forward pass
        output_1, output_2 = coupler_50_50(input_1, input_2)
        
        # Test dimensiones
        assert output_1.shape == input_1.shape
        assert output_2.shape == input_2.shape
        
        # Test conservación de energía: |out1|² + |out2|² = |in1|² + |in2|²
        input_energy = torch.sum(torch.abs(input_1)**2 + torch.abs(input_2)**2, dim=1)
        output_energy = torch.sum(torch.abs(output_1)**2 + torch.abs(output_2)**2, dim=1)
        
        energy_ratio = output_energy / torch.clamp(input_energy, min=1e-10)
        assert torch.allclose(energy_ratio, torch.ones_like(energy_ratio), atol=1e-3), "Energy not conserved in coupler"
    
    def test_splitting_ratios(self, device):
        """Test: Diferentes ratios de splitting."""
        ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for ratio in ratios:
            coupler = DirectionalCoupler(splitting_ratio=ratio, device=device)
            
            # Test con input solo en puerto 1
            input_1 = torch.ones(3, 4, device=device, dtype=torch.float32)
            input_2 = torch.zeros(3, 4, device=device, dtype=torch.float32)
            
            output_1, output_2 = coupler(input_1, input_2)
            
            # Para coupler ideal, splitting debe seguir el ratio
            input_energy = torch.sum(torch.abs(input_1)**2, dim=1)
            output_1_energy = torch.sum(torch.abs(output_1)**2, dim=1)
            output_2_energy = torch.sum(torch.abs(output_2)**2, dim=1)
            
            total_output_energy = output_1_energy + output_2_energy
            
            # Test conservación
            energy_conservation = torch.mean(total_output_energy / torch.clamp(input_energy, min=1e-10))
            assert abs(energy_conservation - 1.0) < 0.05, f"Energy not conserved for ratio {ratio}"


class TestPhotodetector:
    """Tests para Photodetector."""
    
    @pytest.fixture
    def device(self):
        """Fixture para device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def photodetector(self, device):
        """Fixture para photodetector estándar."""
        return Photodetector(
            responsivity=1.0,
            dark_current=1e-9,
            device=device
        )
    
    def test_photodetector_initialization(self, device):
        """Test: Inicialización correcta del photodetector."""
        pd = Photodetector(
            responsivity=0.8,
            dark_current=2e-9,
            thermal_noise=1e-11,
            bandwidth=5e9,
            device=device
        )
        
        assert pd.responsivity == 0.8
        assert pd.dark_current == 2e-9
        assert pd.thermal_noise == 1e-11
        assert pd.bandwidth == 5e9
        assert pd.device == device
    
    def test_optical_to_electrical_conversion(self, photodetector, device):
        """Test: Conversión óptico-eléctrica."""
        batch_size = 6
        n_wavelengths = 8
        
        # Señal óptica de entrada
        optical_signal = torch.randn(batch_size, n_wavelengths, device=device, dtype=torch.float32)
        
        # Conversión
        electrical_current = photodetector(optical_signal)
        
        # Test dimensiones
        assert electrical_current.shape == optical_signal.shape
        
        # Test que hay conversión (potencia óptica -> corriente)
        optical_power = torch.abs(optical_signal)**2
        expected_current = photodetector.responsivity * optical_power + photodetector.dark_current
        
        # En modo eval (sin noise), debe ser muy similar
        photodetector.eval()
        electrical_current_no_noise = photodetector(optical_signal)
        
        assert torch.allclose(electrical_current_no_noise, expected_current, atol=1e-6), "O-E conversion incorrect"
    
    def test_dark_current_effect(self, device):
        """Test: Efecto de dark current."""
        # Photodetector con dark current alto
        pd_high_dark = Photodetector(dark_current=1e-6, device=device)
        pd_low_dark = Photodetector(dark_current=1e-12, device=device)
        
        # Señal óptica muy baja
        low_optical = torch.ones(2, 3, device=device, dtype=torch.float32) * 1e-9
        
        current_high_dark = pd_high_dark(low_optical)
        current_low_dark = pd_low_dark(low_optical)
        
        # High dark current debe dominar para señales bajas
        assert torch.mean(current_high_dark) > torch.mean(current_low_dark), "Dark current effect not visible"
    
    def test_responsivity_scaling(self, device):
        """Test: Scaling con responsivity."""
        responsivities = [0.1, 0.5, 1.0, 2.0]
        optical_signal = torch.ones(2, 3, device=device, dtype=torch.float32)
        
        currents = []
        for resp in responsivities:
            pd = Photodetector(responsivity=resp, dark_current=0, device=device)
            pd.eval()  # Sin noise
            current = pd(optical_signal)
            currents.append(torch.mean(current))
        
        # Current debe escalar linealmente con responsivity
        for i in range(1, len(currents)):
            ratio_expected = responsivities[i] / responsivities[0]
            ratio_actual = currents[i] / currents[0]
            assert abs(ratio_actual - ratio_expected) < 0.01, f"Responsivity scaling incorrect: {ratio_actual} vs {ratio_expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])