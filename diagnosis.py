#!/usr/bin/env python3
"""
Script de Diagnóstico Completo para el Microring - v5.4

Este script ejecuta diagnósticos detallados para identificar
exactamente por qué el extinction ratio está tan bajo.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Añadir ruta del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchonn.layers import MicroringResonator

def diagnostic_microring_resonance():
    """Diagnóstico completo del microring resonator."""
    print("🔍 DIAGNÓSTICO COMPLETO DEL MICRORING v5.4")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Crear microring con parámetros de test
    mrr = MicroringResonator(
        radius=10e-6,
        q_factor=5000,
        center_wavelength=1550e-9,
        coupling_mode="critical",
        device=device
    )
    
    print(f"\n📊 PARÁMETROS DEL MICRORING:")
    print(f"   Radius: {mrr.radius*1e6:.1f} μm")
    print(f"   Q factor: {mrr.q_factor}")
    print(f"   Center wavelength: {mrr.center_wavelength*1e9:.3f} nm")
    print(f"   Alpha: {mrr.alpha:.6f}")
    print(f"   Kappa critical: {mrr.kappa_critical:.6f}")
    print(f"   Kappa usado: {mrr.coupling_strength_target:.6f}")
    print(f"   FSR: {mrr.fsr*1e12:.0f} pm")
    print(f"   ER teórico: {mrr.extinction_ratio_theory_db:.1f} dB")
    
    # Test 1: Verificar rango de wavelengths
    print(f"\n🔍 TEST 1: RANGO DE WAVELENGTHS")
    wavelengths = mrr.get_recommended_wavelengths(1000)
    wl_min, wl_max = torch.min(wavelengths), torch.max(wavelengths)
    wl_center = mrr.center_wavelength
    
    print(f"   Rango: {wl_min*1e9:.3f} - {wl_max*1e9:.3f} nm")
    print(f"   Centro: {wl_center*1e9:.3f} nm")
    print(f"   Rango total: {(wl_max-wl_min)*1e12:.0f} pm")
    print(f"   FWHM esperado: {wl_center/mrr.q_factor*1e12:.0f} pm")
    
    # Verificar que el centro está en el rango
    if wl_min <= wl_center <= wl_max:
        print(f"   ✅ Centro wavelength en el rango")
    else:
        print(f"   ❌ Centro wavelength FUERA del rango")
        return False
    
    # Test 2: Respuesta espectral detallada
    print(f"\n🔍 TEST 2: RESPUESTA ESPECTRAL")
    input_signal = torch.ones(1, 1000, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = mrr(input_signal, wavelengths)
    
    through_response = output['through'][0]
    drop_response = output['drop'][0]
    
    print(f"   Through min: {torch.min(through_response):.6f}")
    print(f"   Through max: {torch.max(through_response):.6f}")
    print(f"   Through mean: {torch.mean(through_response):.6f}")
    print(f"   Through std: {torch.std(through_response):.6f}")
    
    print(f"   Drop min: {torch.min(drop_response):.6f}")
    print(f"   Drop max: {torch.max(drop_response):.6f}")
    print(f"   Drop mean: {torch.mean(drop_response):.6f}")
    
    # Test 3: Conservación de energía
    print(f"\n🔍 TEST 3: CONSERVACIÓN DE ENERGÍA")
    total_energy = through_response + drop_response
    energy_min, energy_max = torch.min(total_energy), torch.max(total_energy)
    energy_mean = torch.mean(total_energy)
    
    print(f"   Energía total min: {energy_min:.6f}")
    print(f"   Energía total max: {energy_max:.6f}")
    print(f"   Energía total mean: {energy_mean:.6f}")
    print(f"   Energía esperada: ~{mrr.alpha:.3f} (α)")
    
    if energy_max > 1.01:
        print(f"   ❌ Violación de conservación de energía: {energy_max:.6f} > 1.0")
        return False
    else:
        print(f"   ✅ Conservación de energía OK")
    
    # Test 4: Detección de resonancia
    print(f"\n🔍 TEST 4: DETECCIÓN DE RESONANCIA")
    min_idx = torch.argmin(through_response)
    max_idx = torch.argmax(through_response)
    
    resonance_wl = wavelengths[min_idx]
    off_resonance_wl = wavelengths[max_idx]
    
    print(f"   Resonancia encontrada en: {resonance_wl*1e9:.3f} nm (índice {min_idx})")
    print(f"   Off-resonance en: {off_resonance_wl*1e9:.3f} nm (índice {max_idx})")
    print(f"   Error resonancia: {abs(resonance_wl - wl_center)*1e12:.1f} pm")
    
    # Verificar que la resonancia está cerca del centro
    resonance_error = abs(resonance_wl - wl_center)
    fwhm = wl_center / mrr.q_factor
    
    if resonance_error < fwhm:
        print(f"   ✅ Resonancia bien centrada (error < FWHM)")
    else:
        print(f"   ❌ Resonancia descentrada (error = {resonance_error/fwhm:.1f} × FWHM)")
        return False
    
    # Test 5: Cálculo de extinction ratio
    print(f"\n🔍 TEST 5: EXTINCTION RATIO")
    min_transmission = through_response[min_idx]
    max_transmission = through_response[max_idx]
    
    print(f"   Min transmission: {min_transmission:.6f}")
    print(f"   Max transmission: {max_transmission:.6f}")
    
    if min_transmission > 1e-12 and max_transmission > min_transmission:
        er_ratio = max_transmission / min_transmission
        er_db = 10 * torch.log10(er_ratio)
        
        print(f"   ER ratio: {er_ratio:.3f}")
        print(f"   ER medido: {er_db:.1f} dB")
        print(f"   ER teórico: {mrr.extinction_ratio_theory_db:.1f} dB")
        print(f"   Error ER: {abs(er_db - mrr.extinction_ratio_theory_db):.1f} dB")
        
        if er_db > 3:
            print(f"   ✅ ER en rango físico")
            return True
        else:
            print(f"   ❌ ER demasiado bajo: {er_db:.1f} dB < 3 dB")
            return False
    else:
        print(f"   ❌ No se puede calcular ER válido")
        return False

def diagnostic_phase_sweep():
    """Diagnóstico de barrido de fase para encontrar la resonancia óptima."""
    print(f"\n🔍 DIAGNÓSTICO DE BARRIDO DE FASE")
    print("-" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear microring
    mrr = MicroringResonator(
        radius=10e-6,
        q_factor=5000,
        center_wavelength=1550e-9,
        coupling_mode="critical",
        device=device
    )
    
    # Wavelengths en un rango pequeño alrededor del centro
    center_wl = mrr.center_wavelength
    wavelengths = torch.linspace(center_wl - 500e-12, center_wl + 500e-12, 100, device=device, dtype=torch.float32)
    input_signal = torch.ones(1, 100, device=device, dtype=torch.float32)
    
    # Probar diferentes valores de phase_shift
    phase_shifts = torch.linspace(-np.pi, np.pi, 20)
    best_er = 0
    best_phase = 0
    
    print(f"Probando {len(phase_shifts)} valores de phase shift...")
    
    for i, phase in enumerate(phase_shifts):
        # Modificar phase_shift temporalmente
        original_phase = mrr.phase_shift.data.clone()
        mrr.phase_shift.data.fill_(phase)
        
        with torch.no_grad():
            output = mrr(input_signal, wavelengths)
        
        through_response = output['through'][0]
        min_trans = torch.min(through_response)
        max_trans = torch.max(through_response)
        
        if min_trans > 1e-12:
            er_ratio = max_trans / min_trans
            er_db = 10 * torch.log10(er_ratio)
            
            if er_db > best_er:
                best_er = er_db
                best_phase = phase
        
        # Restaurar phase_shift original
        mrr.phase_shift.data = original_phase
        
        if i % 5 == 0:
            print(f"   Phase {phase:.3f}: ER = {er_db:.1f} dB")
    
    print(f"\n📊 RESULTADO BARRIDO DE FASE:")
    print(f"   Mejor phase shift: {best_phase:.3f}")
    print(f"   Mejor ER: {best_er:.1f} dB")
    print(f"   Phase actual: {mrr.phase_shift.item():.3f}")
    
    if best_er > 5:
        print(f"   ✅ Se encontró una configuración con ER > 5 dB")
        print(f"   💡 Sugerencia: Ajustar phase_shift a {best_phase:.3f}")
        return True
    else:
        print(f"   ❌ No se encontró configuración con ER alto")
        return False

def plot_spectral_response(save_plot=True):
    """Graficar respuesta espectral para análisis visual."""
    print(f"\n📊 GENERANDO GRÁFICO DE RESPUESTA ESPECTRAL")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mrr = MicroringResonator(
        radius=10e-6,
        q_factor=5000,
        center_wavelength=1550e-9,
        coupling_mode="critical",
        device=device
    )
    
    wavelengths = mrr.get_recommended_wavelengths(2000)
    input_signal = torch.ones(1, 2000, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = mrr(input_signal, wavelengths)
    
    through_response = output['through'][0].cpu().numpy()
    drop_response = output['drop'][0].cpu().numpy()
    wavelengths_nm = wavelengths.cpu().numpy() * 1e9
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths_nm, through_response, 'b-', linewidth=2, label='Through Port')
    plt.axvline(mrr.center_wavelength*1e9, color='red', linestyle='--', alpha=0.7, label='Center Wavelength')
    plt.ylabel('Transmission')
    plt.title(f'Microring Spectral Response - DIAGNOSTIC\nQ={mrr.q_factor}, ER_theory={mrr.extinction_ratio_theory_db:.1f}dB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.subplot(2, 1, 2)
    plt.plot(wavelengths_nm, drop_response, 'r-', linewidth=2, label='Drop Port')
    plt.axvline(mrr.center_wavelength*1e9, color='red', linestyle='--', alpha=0.7, label='Center Wavelength')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('microring_diagnostic.png', dpi=150, bbox_inches='tight')
        print(f"   Gráfico guardado como 'microring_diagnostic.png'")
    
    try:
        plt.show()
    except:
        print("   No se puede mostrar gráfico (entorno sin display)")

def main():
    """Función principal de diagnóstico."""
    print("🚀 INICIANDO DIAGNÓSTICO COMPLETO")
    
    success = True
    
    # Ejecutar diagnósticos
    success &= diagnostic_microring_resonance()
    success &= diagnostic_phase_sweep()
    
    # Generar gráfico
    plot_spectral_response()
    
    # Resultado final
    print(f"\n" + "="*60)
    if success:
        print(f"✅ DIAGNÓSTICO COMPLETO: Microring funcionando correctamente")
    else:
        print(f"❌ DIAGNÓSTICO COMPLETO: Problemas detectados en el microring")
        print(f"\n💡 RECOMENDACIONES:")
        print(f"   1. Verificar que efectos no lineales estén deshabilitados")
        print(f"   2. Ajustar phase_shift para centrar resonancia")
        print(f"   3. Verificar normalización en get_transmission()")
        print(f"   4. Revisar cálculo de extinction ratio en el test")
    
    return success

if __name__ == "__main__":
    main()