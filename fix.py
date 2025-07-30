#!/usr/bin/env python3
"""
🔧 Smart Fix for CoherentONN - Handles Current State

PROBLEMA: Los scripts anteriores ya modificaron el archivo
SOLUCIÓN: Detectar estado actual y aplicar cambios apropiados

ESTRATEGIA:
1. Leer archivo actual y detectar qué cambios ya se aplicaron
2. Aplicar solo los fixes necesarios basándose en el contenido actual
3. Restaurar desde backup si es necesario

USO:
    python smart_fix_nan.py
"""

import os
import shutil
import re
from datetime import datetime
from pathlib import Path

def create_backup(file_path):
    """Crear backup de seguridad del archivo."""
    backup_path = f"{file_path}.smart_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"   📦 Backup created: {backup_path}")
    return backup_path

def analyze_current_state(content):
    """Analizar el estado actual del archivo para saber qué cambios aplicar."""
    state = {
        'has_aggressive_tanh': 'torch.tanh(x * 0.1)' in content,
        'has_torch_pow_04': 'torch.pow(activated_stable, 0.4)' in content or 'torch.pow(optical_intensity, 0.4)' in content,
        'has_large_epsilon': 'epsilon = 1e-3' in content,
        'has_gradient_clipping': 'torch.nan_to_num' in content,
        'has_nan_protection': 'electrical_output = torch.nan_to_num(electrical_output' in content
    }
    return state

def apply_smart_fixes(content, state):
    """Aplicar fixes basándose en el estado actual del archivo."""
    fixes_applied = []
    
    # Fix 1: Revertir torch.pow problemático (sin importar dónde esté)
    if state['has_torch_pow_04']:
        print("   🔧 Revirtiendo torch.pow(x, 0.4) problemático...")
        # Buscar y reemplazar cualquier torch.pow con 0.4
        content = re.sub(r'torch\.pow\([^,]+,\s*0\.4\)', 
                        lambda m: m.group(0).replace('torch.pow(', 'torch.sqrt(').replace(', 0.4)', ')'),
                        content)
        fixes_applied.append("torch.pow → torch.sqrt")
    
    # Fix 2: Revertir tanh normalization agresiva
    if state['has_aggressive_tanh']:
        print("   🔧 Revirtiendo tanh normalization agresiva...")
        # Reemplazar la sección completa de tanh
        tanh_pattern = r'x_normalized = torch\.tanh\(x \* 0\.1\).*?x_scaled = x_positive \* 0\.5'
        if re.search(tanh_pattern, content, re.DOTALL):
            replacement = '''x_abs = torch.abs(x)  # Ensure non-negative
        x_max = torch.max(x_abs) + 1e-4  # Avoid division by zero  
        x_normalized = x_abs / x_max  # Simple max normalization'''
            content = re.sub(tanh_pattern, replacement, content, flags=re.DOTALL)
            fixes_applied.append("tanh → max normalization")
    
    # Fix 3: Reducir epsilon si es demasiado grande
    if state['has_large_epsilon']:
        print("   🔧 Reduciendo epsilon demasiado grande...")
        content = content.replace('epsilon = 1e-3', 'epsilon = 1e-4')
        fixes_applied.append("epsilon 1e-3 → 1e-4")
    
    # Fix 4: Agregar protección NaN en electrical_output si no existe
    if not state['has_nan_protection']:
        print("   🔧 Agregando protección NaN en electrical_output...")
        # Buscar la línea de logits y agregar protección antes
        logits_pattern = r'(\s+)logits = self\.final_layer\(electrical_output\)'
        if re.search(logits_pattern, content):
            replacement = r'\1# Proteger contra NaN/Inf\n\1electrical_output = torch.clamp(electrical_output, min=0.0, max=10.0)\n\1electrical_output = torch.nan_to_num(electrical_output, nan=0.0)\n\1logits = self.final_layer(electrical_output)'
            content = re.sub(logits_pattern, replacement, content)
            fixes_applied.append("NaN protection added")
    
    # Fix 5: Mejorar gradient clipping si existe
    if state['has_gradient_clipping']:
        print("   🔧 Mejorando gradient clipping...")
        # Hacer clipping más conservador
        content = content.replace('posinf=1.0, neginf=-1.0', 'posinf=0.1, neginf=-0.1')
        fixes_applied.append("gradient clipping improved")
    
    # Fix 6: Agregar bounds más conservadores en activated_stable/activated_safe
    if 'activated_stable = torch.clamp(activated, min=epsilon, max=100.0)' in content:
        content = content.replace('activated_stable = torch.clamp(activated, min=epsilon, max=100.0)',
                                'activated_safe = torch.clamp(activated, min=1e-4, max=10.0)')
        fixes_applied.append("clamp bounds conservative")
    
    return content, fixes_applied

def restore_from_backup_if_needed():
    """Ofrecer restaurar desde backup si el archivo está muy corrupto."""
    file_path = "torchonn/onns/architectures/coherent_onn.py"
    
    # Buscar backups disponibles
    backups = []
    for f in os.listdir("torchonn/onns/architectures/"):
        if f.startswith("coherent_onn.py.") and ("backup" in f or "fix" in f):
            backups.append(f)
    
    if backups:
        print(f"\n📦 Backups disponibles:")
        for i, backup in enumerate(sorted(backups, reverse=True)[:5]):  # Mostrar solo los 5 más recientes
            print(f"   {i+1}. {backup}")
        
        choice = input(f"\n❓ ¿Restaurar desde backup? (número/n): ").strip().lower()
        
        if choice.isdigit() and 1 <= int(choice) <= len(backups):
            backup_to_restore = backups[int(choice)-1]
            backup_path = f"torchonn/onns/architectures/{backup_to_restore}"
            shutil.copy2(backup_path, file_path)
            print(f"   ✅ Archivo restaurado desde: {backup_to_restore}")
            return True
    
    return False

def main():
    """Función principal."""
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print("🌟      SMART FIX - ANALIZA ESTADO ACTUAL      🌟")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    
    file_path = "torchonn/onns/architectures/coherent_onn.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} no encontrado")
        return 1
    
    # Leer archivo actual
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Analizar estado actual
    print("🔍 Analizando estado actual del archivo...")
    state = analyze_current_state(content)
    
    print("📊 Estado detectado:")
    for key, value in state.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}: {value}")
    
    # Verificar si el archivo está muy corrupto
    has_major_issues = state['has_aggressive_tanh'] or state['has_torch_pow_04']
    
    if has_major_issues:
        print(f"\n⚠️ Detectados cambios problemáticos que pueden causar NaN")
        choice = input(f"¿Intentar fix automático (f) o restaurar desde backup (b)? [f/b]: ").strip().lower()
        
        if choice == 'b':
            if restore_from_backup_if_needed():
                print(f"✅ Archivo restaurado. Ejecutar: python demos/demo_onn.py --quick")
                return 0
            else:
                print(f"❌ No se pudo restaurar desde backup")
                return 1
    
    # Crear backup del estado actual
    backup_path = create_backup(file_path)
    
    # Aplicar fixes inteligentes
    print(f"\n🔧 Aplicando fixes basándose en estado actual...")
    modified_content, fixes_applied = apply_smart_fixes(content, state)
    
    # Escribir archivo corregido
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"\n🎉 SMART FIXES APLICADOS:")
    for fix in fixes_applied:
        print(f"   ✅ {fix}")
    
    if fixes_applied:
        print(f"\n🚀 TEST RECOMENDADO:")
        print(f"   python demos/demo_onn.py --quick")
        print(f"   # Verificar que no hay más ❌ Failed: NaN in output")
        
        print(f"\n💡 SI AÚN HAY PROBLEMAS:")
        print(f"   Ejecutar nuevamente con opción de restaurar backup")
        return 0
    else:
        print(f"\n💭 No se aplicaron fixes - archivo parece estar en buen estado")
        print(f"   Si hay problemas, pueden ser en otra parte del código")
        return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)