#!/usr/bin/env python3
"""
🔧 Corrección de contexto de indentación
El problema está en líneas ANTES de la 193
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Crear backup seguro del archivo."""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backup creado: {backup_path}")
        return backup_path
    return None

def analyze_context_thoroughly():
    """Analizar contexto completo alrededor de línea 193"""
    filepath = "examples/Example1.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    problem_line = 193
    
    print(f"🔍 ANÁLISIS COMPLETO - Líneas {problem_line-10} a {problem_line+5}:")
    print("=" * 80)
    
    for i in range(max(0, problem_line-11), min(len(lines), problem_line+6)):
        line = lines[i]
        line_content = line.rstrip('\n')
        
        # Detectar tabs vs espacios
        leading_whitespace = len(line) - len(line.lstrip())
        has_tabs = '\t' in line[:leading_whitespace]
        space_count = line[:leading_whitespace].count(' ')
        tab_count = line[:leading_whitespace].count('\t')
        
        # Marcar línea problemática
        marker = " <<<< PROBLEMA" if i == problem_line - 1 else ""
        
        print(f"{i+1:3d}: {repr(line_content)}{marker}")
        print(f"     Espacios: {space_count}, Tabs: {tab_count}, Total: {leading_whitespace}")
        
        # Detectar problemas potenciales
        if has_tabs and space_count > 0:
            print(f"     ⚠️ MEZCLA DE TABS Y ESPACIOS")
        
        if line.rstrip().endswith(':') and i < len(lines) - 1:
            next_line = lines[i + 1]
            next_leading = len(next_line) - len(next_line.lstrip())
            if next_leading <= leading_whitespace:
                print(f"     ⚠️ SIGUIENTE LÍNEA NO TIENE MÁS INDENTACIÓN")
        
        print()
    
    return lines

def fix_indentation_context():
    """Corregir problemas de indentación en el contexto"""
    filepath = "examples/Example1.py"
    
    backup_file(filepath)
    
    lines = analyze_context_thoroughly()
    if not lines:
        return False
    
    problem_line = 193
    fixed = False
    
    # Buscar y corregir problemas específicos
    for i in range(max(0, problem_line-10), min(len(lines), problem_line+5)):
        line = lines[i]
        
        # 1. Convertir tabs a espacios
        if '\t' in line:
            lines[i] = line.expandtabs(4)
            print(f"✅ Línea {i+1}: Tabs convertidos a espacios")
            fixed = True
        
        # 2. Buscar líneas que deberían tener : al final
        if i < len(lines) - 1:
            stripped = line.strip()
            next_line = lines[i + 1].strip()
            
            # Si línea actual es control flow sin : y siguiente está indentada
            if (stripped.startswith(('if ', 'elif ', 'else', 'for ', 'while ', 'try', 'except', 'finally', 'with ', 'def ', 'class ')) 
                and not stripped.endswith(':') 
                and next_line 
                and len(lines[i + 1]) - len(lines[i + 1].lstrip()) > len(line) - len(line.lstrip())):
                
                lines[i] = line.rstrip() + ':\n'
                print(f"✅ Línea {i+1}: Agregados dos puntos (:)")
                fixed = True
        
        # 3. Corregir indentación incorrecta específica
        if i == problem_line - 1:  # Línea 193
            # Verificar línea anterior (192)
            if i > 0:
                prev_line = lines[i-1]
                curr_line = lines[i]
                
                # Si línea anterior termina en : y actual no tiene más indentación
                if prev_line.strip().endswith(':'):
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    curr_indent = len(curr_line) - len(curr_line.lstrip())
                    
                    if curr_indent <= prev_indent:
                        # Agregar 4 espacios más
                        lines[i] = ' ' * (prev_indent + 4) + curr_line.lstrip()
                        print(f"✅ Línea {i+1}: Indentación incrementada después de :")
                        fixed = True
    
    if fixed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("\n✅ Correcciones aplicadas")
        return True
    
    return False

def manual_fix_around_193():
    """Corrección manual específica alrededor de línea 193"""
    filepath = "examples/Example1.py"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar el patrón específico problemático
    # Es probable que sea el debug code que agregamos antes
    
    # Patrón 1: Debug code mal indentado
    problematic_pattern = r'\n        # 🔍 DEBUG Simple: Parámetros del microring\n        kappa_value = mrr\.coupling_tuning\.item\(\)\n        print\(f"   🔍 Debug - κ: \{kappa_value:.4f\}, Q: \{mrr\.q_factor\}"\)\n\n        print\(f"📊 Resultados Microring:"\)'
    
    if problematic_pattern in content:
        # Remover el debug problemático completamente
        content = re.sub(problematic_pattern, '\n        print(f"📊 Resultados Microring:")', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Debug problemático removido completamente")
        return True
    
    # Patrón 2: Bloque de código mal indentado
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Wavelength central:' in line and i == 192:  # Línea 193 (índice 192)
            # Verificar líneas anteriores
            for j in range(max(0, i-5), i):
                if lines[j].strip() and not lines[j].startswith('        '):
                    # Corregir indentación de líneas anteriores
                    lines[j] = '        ' + lines[j].lstrip()
                    print(f"✅ Línea {j+1}: Indentación corregida")
            
            # Escribir archivo corregido
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True
    
    return False

def validate_syntax_after_fix():
    """Validar sintaxis después de corrección"""
    filepath = "examples/Example1.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        compile(source, filepath, 'exec')
        print("✅ SINTAXIS CORRECTA")
        return True
    
    except SyntaxError as e:
        print(f"❌ Error persiste en línea {e.lineno}: {e.msg}")
        if e.text:
            print(f"   Texto: {e.text.strip()}")
        return False

def main():
    """Función principal de corrección de contexto"""
    print("🔧 CORRECCIÓN DE CONTEXTO DE INDENTACIÓN")
    print("=" * 60)
    
    success = False
    
    # 1. Análisis detallado
    print("\n🔍 PASO 1: Análisis detallado del contexto")
    analyze_context_thoroughly()
    
    # 2. Corrección de contexto
    print("\n🔧 PASO 2: Corrección de contexto")
    if fix_indentation_context():
        success = True
    
    # 3. Corrección manual si es necesario
    if not success:
        print("\n🔧 PASO 3: Corrección manual específica")
        if manual_fix_around_193():
            success = True
    
    # 4. Validación final
    print("\n✅ PASO 4: Validación final")
    syntax_ok = validate_syntax_after_fix()
    
    if syntax_ok:
        print("\n🎯 ¡PROBLEMA RESUELTO!")
        print("   python examples/Example1.py")
    else:
        print("\n❌ PROBLEMA PERSISTE")
        print("   Usar backup más reciente:")
        print("   ls examples/Example1.py.backup_*")

if __name__ == "__main__":
    main()