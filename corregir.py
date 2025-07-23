#!/usr/bin/env python3
"""
🔍 INVESTIGADOR DE PROBLEMAS PROFUNDOS + SOLUCIÓN FINAL

SITUACIÓN: Todas las correcciones están aplicadas ✅, pero el problema persiste ❌
- Parámetros correctos: κ=0.2, Q=15000, range=1000e-12 ✅
- Función get_transmission corregida ✅  
- Gráfica sigue mostrando Through > 4.0 ❌

HIPÓTESIS DE PROBLEMAS OCULTOS:
1. 🔄 Caché de Python/importaciones obsoletas
2. 🧮 Error en el cálculo o aplicación de la transmisión
3. 📊 Problema en el plotting/scaling
4. 🔧 Función get_transmission no se está usando
5. 🎯 Problema en otra parte del pipeline

Este script va a investigar sistemáticamente cada posibilidad.
"""

import os
import sys
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DeeperIssueInvestigator:
    """Investigador de problemas profundos."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%H%M%S')
        self.findings = {}
        
    def log(self, message, level="INFO"):
        """Log con timestamp."""
        symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🚨"}
        print(f"{symbols.get(level, 'ℹ️')} {message}")
    
    def force_clean_import(self):
        """Forzar reimportación limpia de todos los módulos."""
        self.log("FORZANDO REIMPORTACIÓN LIMPIA DE MÓDULOS", "INFO")
        print("-" * 60)
        
        # Limpiar módulos de torchonn del caché
        modules_to_clear = []
        for module_name in sys.modules.keys():
            if 'torchonn' in module_name:
                modules_to_clear.append(module_name)
        
        for module_name in modules_to_clear:
            del sys.modules[module_name]
            self.log(f"Módulo limpiado del caché: {module_name}")
        
        # Asegurar que el directorio actual está en el path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            self.log(f"Directorio agregado al path: {current_dir}")
        
        # Reimportar con forzado
        try:
            import torch
            from torchonn.layers.microring import MicroringResonator
            self.log("✅ Módulos reimportados exitosamente", "SUCCESS")
            return True
        except Exception as e:
            self.log(f"❌ Error reimportando módulos: {e}", "ERROR")
            return False
    
    def test_get_transmission_directly(self):
        """Probar la función get_transmission directamente."""
        self.log("PROBANDO FUNCIÓN get_transmission DIRECTAMENTE", "INFO")
        print("-" * 60)
        
        try:
            from torchonn.layers.microring import MicroringResonator
            
            device = torch.device("cpu")
            
            # Crear microring con parámetros que sabemos que están correctos
            mrr = MicroringResonator(
                radius=10e-6,
                coupling_strength=0.2,
                q_factor=15000,
                center_wavelength=1550e-9,
                device=device
            )
            
            self.log(f"Microring creado: κ={mrr.coupling_strength:.3f}, Q={mrr.q_factor}")
            
            # Test directo de get_transmission
            wavelengths = torch.linspace(1549.5e-9, 1550.5e-9, 100, device=device)
            
            # Llamar directamente a get_transmission
            through_trans, drop_trans = mrr.get_transmission(wavelengths)
            
            self.log(f"RESULTADO DIRECTO DE get_transmission:")
            self.log(f"  Through range: {torch.min(through_trans):.4f} - {torch.max(through_trans):.4f}")
            self.log(f"  Drop range: {torch.min(drop_trans):.4f} - {torch.max(drop_trans):.4f}")
            
            # Validar rangos físicos
            if torch.max(through_trans) > 1.01:
                self.log("🚨 CRÍTICO: get_transmission da Through > 1.0", "CRITICAL")
                return False, through_trans, drop_trans
            else:
                self.log("✅ get_transmission da valores físicos válidos", "SUCCESS")
                return True, through_trans, drop_trans
                
        except Exception as e:
            self.log(f"❌ Error probando get_transmission: {e}", "ERROR")
            return False, None, None
    
    def test_full_forward_pass(self):
        """Probar el forward pass completo del microring."""
        self.log("PROBANDO FORWARD PASS COMPLETO", "INFO")
        print("-" * 60)
        
        try:
            from torchonn.layers.microring import MicroringResonator
            
            device = torch.device("cpu")
            
            mrr = MicroringResonator(
                radius=10e-6,
                coupling_strength=0.2,
                q_factor=15000,
                center_wavelength=1550e-9,
                device=device
            )
            
            # Test forward pass completo
            wavelengths = torch.linspace(1549.5e-9, 1550.5e-9, 100, device=device)
            input_signal = torch.ones(1, 100, device=device)
            
            with torch.no_grad():
                output = mrr(input_signal, wavelengths)
                through_output = output['through'][0]
                drop_output = output['drop'][0]
            
            self.log(f"RESULTADO FORWARD PASS COMPLETO:")
            self.log(f"  Through output: {torch.min(through_output):.4f} - {torch.max(through_output):.4f}")
            self.log(f"  Drop output: {torch.min(drop_output):.4f} - {torch.max(drop_output):.4f}")
            
            # Comparar con get_transmission directo
            through_trans, drop_trans = mrr.get_transmission(wavelengths)
            
            # Verificar si son iguales (deberían serlo si input=1)
            diff_through = torch.max(torch.abs(through_output - through_trans))
            diff_drop = torch.max(torch.abs(drop_output - drop_trans))
            
            self.log(f"DIFERENCIA entre forward y get_transmission:")
            self.log(f"  Through diff: {diff_through:.6f}")
            self.log(f"  Drop diff: {diff_drop:.6f}")
            
            if diff_through > 1e-5 or diff_drop > 1e-5:
                self.log("⚠️ Forward pass difiere de get_transmission", "WARNING")
            else:
                self.log("✅ Forward pass consistente con get_transmission", "SUCCESS")
            
            return through_output, drop_output, wavelengths
            
        except Exception as e:
            self.log(f"❌ Error en forward pass: {e}", "ERROR")
            return None, None, None
    
    def investigate_demo2_method(self):
        """Investigar específicamente el método demo_2_microring_spectral_response."""
        self.log("INVESTIGANDO MÉTODO demo_2_microring_spectral_response", "INFO")
        print("-" * 60)
        
        try:
            # Ejecutar solo la parte relevante del demo
            sys.path.insert(0, '.')
            
            # Simular el demo_2 exactamente como está en Example1.py
            from torchonn.layers.microring import MicroringResonator
            
            device = torch.device("cpu")
            
            # Recrear exactamente como en Example1.py
            mrr = MicroringResonator(
                radius=10e-6,
                coupling_strength=0.2,   # ✅ Corregido
                q_factor=15000,          # ✅ Corregido  
                center_wavelength=1550e-9,
                device=device
            )
            
            # Wavelength sweep exactamente como en Example1.py
            n_points = 1000
            wavelength_range = 1000e-12  # ✅ Corregido
            wavelengths = torch.linspace(
                1550e-9 - wavelength_range/2, 
                1550e-9 + wavelength_range/2, 
                n_points, 
                device=device
            )
            
            # Input signal exactamente como en Example1.py
            input_signal = torch.ones(1, n_points, device=device)
            
            # Simular respuesta exactamente como en Example1.py
            with torch.no_grad():
                output = mrr(input_signal, wavelengths)
                through_response = output['through'][0]
                drop_response = output['drop'][0]
            
            self.log(f"RESULTADO SIMULANDO demo_2 EXACTAMENTE:")
            self.log(f"  Through: {torch.min(through_response):.4f} - {torch.max(through_response):.4f}")
            self.log(f"  Drop: {torch.min(drop_response):.4f} - {torch.max(drop_response):.4f}")
            
            # Calcular extinction ratio
            min_idx = torch.argmin(through_response)
            max_through = torch.max(through_response)
            min_through = through_response[min_idx]
            
            if min_through > 1e-10:
                extinction_ratio_db = 10 * torch.log10(max_through / min_through)
                self.log(f"  Extinction ratio: {extinction_ratio_db:.1f} dB")
            else:
                self.log(f"  Extinction ratio: ∞ dB (min_through ≈ 0)")
            
            # Crear gráfico de diagnóstico
            self.create_diagnostic_plot(wavelengths, through_response, drop_response)
            
            return through_response, drop_response, wavelengths
            
        except Exception as e:
            self.log(f"❌ Error investigando demo_2: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def create_diagnostic_plot(self, wavelengths, through_response, drop_response):
        """Crear gráfico de diagnóstico."""
        try:
            wavelengths_nm = wavelengths.cpu().numpy() * 1e9
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Through port
            ax1.plot(wavelengths_nm, through_response.cpu().numpy(), 'b-', linewidth=2, label='Through Port DIAGNOSTICO')
            ax1.set_ylabel('Transmission')
            ax1.set_title('DIAGNÓSTICO: Microring Response')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, max(1.1, torch.max(through_response).item() * 1.1)])
            
            # Drop port
            ax2.plot(wavelengths_nm, drop_response.cpu().numpy(), 'r-', linewidth=2, label='Drop Port DIAGNOSTICO')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Transmission') 
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, max(1.1, torch.max(drop_response).item() * 1.1)])
            
            # Estadísticas
            stats_text = f'''DIAGNÓSTICO {self.timestamp}:
Through: {torch.min(through_response):.4f} - {torch.max(through_response):.4f}
Drop: {torch.min(drop_response):.4f} - {torch.max(drop_response):.4f}
Max Through: {"✅ OK" if torch.max(through_response) <= 1.01 else "❌ > 1.0"}'''
            
            ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plot_filename = f'diagnostic_plot_{self.timestamp}.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.show()
            
            self.log(f"📊 Gráfico diagnóstico guardado: {plot_filename}")
            
        except Exception as e:
            self.log(f"Error creando gráfico diagnóstico: {e}", "WARNING")
    
    def run_deep_investigation(self):
        """Ejecutar investigación profunda completa."""
        self.log("🚀 INICIANDO INVESTIGACIÓN PROFUNDA DE PROBLEMAS OCULTOS", "INFO")
        print("=" * 100)
        
        # Paso 1: Limpiar caché e reimportar
        self.log("\n1️⃣ LIMPIANDO CACHÉ Y REIMPORTANDO")
        if not self.force_clean_import():
            self.log("❌ Fallo en reimportación - problema crítico", "CRITICAL")
            return False
        
        # Paso 2: Test directo de get_transmission
        self.log("\n2️⃣ TEST DIRECTO DE get_transmission")
        transmission_ok, through_trans, drop_trans = self.test_get_transmission_directly()
        
        if not transmission_ok:
            self.log("🚨 PROBLEMA ENCONTRADO: get_transmission da valores > 1.0", "CRITICAL")
            self.log("🔧 SOLUCIÓN: La función get_transmission AÚN tiene errores", "WARNING")
            return False
        
        # Paso 3: Test de forward pass completo
        self.log("\n3️⃣ TEST DE FORWARD PASS COMPLETO")
        through_out, drop_out, wavelengths = self.test_full_forward_pass()
        
        if through_out is None:
            self.log("❌ Fallo en forward pass", "ERROR")
            return False
        
        # Paso 4: Investigar demo_2 específicamente
        self.log("\n4️⃣ INVESTIGANDO DEMO_2 ESPECÍFICAMENTE")
        demo_through, demo_drop, demo_wavelengths = self.investigate_demo2_method()
        
        if demo_through is None:
            self.log("❌ Fallo investigando demo_2", "ERROR")
            return False
        
        # Paso 5: Análisis final
        self.log("\n5️⃣ ANÁLISIS FINAL")
        max_demo_through = torch.max(demo_through)
        
        if max_demo_through > 1.01:
            self.log(f"🚨 PROBLEMA CONFIRMADO: demo_2 da Through = {max_demo_through:.4f} > 1.0", "CRITICAL")
            self.log("🔧 La función get_transmission AÚN necesita corrección", "WARNING")
            return False
        else:
            self.log(f"✅ PROBLEMA RESUELTO: demo_2 da Through = {max_demo_through:.4f} ≤ 1.0", "SUCCESS")
            
            # Calcular extinction ratio
            min_through = torch.min(demo_through)
            if min_through > 1e-10:
                extinction_ratio = 10 * torch.log10(max_demo_through / min_through)
                self.log(f"✅ Extinction ratio: {extinction_ratio:.1f} dB", "SUCCESS")
                
                if extinction_ratio > 15:
                    self.log("🎉 ¡PROBLEMA COMPLETAMENTE RESUELTO!", "SUCCESS")
                    return True
                else:
                    self.log("⚠️ Extinction ratio aún bajo pero físicamente válido", "WARNING")
                    return True
            else:
                self.log("✅ Through mínimo ≈ 0 (excelente extinction)", "SUCCESS")
                return True

def main():
    """Función principal."""
    investigator = DeeperIssueInvestigator()
    success = investigator.run_deep_investigation()
    
    if success:
        print("\n" + "🎉" * 40)
        print("¡PROBLEMA RESUELTO! El microring ahora funciona correctamente.")
        print("🎯 EJECUTA: python examples/Example1.py")
        print("📊 DEBERÍAS VER: Through ∈ [0,1], Extinction ratio > 15 dB")
        print("🎉" * 40)
    else:
        print("\n" + "❌" * 40)  
        print("PROBLEMA PERSISTE - get_transmission aún tiene errores")
        print("🔍 REVISAR logs arriba para más detalles")
        print("❌" * 40)
        
        # Información adicional para debugging
        print("\n🔧 INFORMACIÓN ADICIONAL:")
        print("- Todos los parámetros están correctos ✅")
        print("- La función get_transmission parece correcta en el código ✅")
        print("- Pero sigue dando valores > 1.0 ❌")
        print("- Posible problema: indentación, sintaxis, o lógica oculta")
    
    return success

if __name__ == "__main__":
    main()