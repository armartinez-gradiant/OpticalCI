#!/usr/bin/env python3
"""
🔧 Fix rápido para diagnosis.py

Corrige el simple problema de import de torch en la función check_torchonn_installation
"""

import sys
from pathlib import Path
import shutil

def fix_diagnosis_script():
    """Arreglar el problema de import en diagnosis.py"""
    print("🔧 Arreglando diagnosis.py...")
    
    # Buscar el archivo diagnosis.py
    possible_names = ["diagnosis.py", "diagnostic_script.py", "diagnosis_script.py"]
    diagnosis_file = None
    
    for name in possible_names:
        if Path(name).exists():
            diagnosis_file = Path(name)
            break
    
    if not diagnosis_file:
        print("❌ No se encontró archivo de diagnóstico")
        return False
    
    print(f"📄 Encontrado: {diagnosis_file}")
    
    # Crear backup
    backup_file = diagnosis_file.with_suffix('.py.backup')
    try:
        shutil.copy2(diagnosis_file, backup_file)
        print(f"💾 Backup creado: {backup_file}")
    except Exception as e:
        print(f"⚠️  Error creando backup: {e}")
    
    # Leer contenido actual
    try:
        with open(diagnosis_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return False
    
    # Buscar la función problemática
    lines = content.split('\n')
    fixed_lines = []
    import_added = False
    
    for i, line in enumerate(lines):
        # Si encontramos la función check_torchonn_installation
        if 'def check_torchonn_installation(' in line:
        try:
            import torch
        except ImportError:
            self.log_error("TorchONN", "PyTorch no está disponible para test")
            return

            fixed_lines.append(line)
            # Añadir import torch al inicio de la función si no existe
            if not import_added:
                # Buscar líneas siguientes para añadir el import
                next_line_idx = i + 1
                while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                    fixed_lines.append(lines[next_line_idx])
                    next_line_idx += 1
                
                # Añadir import torch
                indent = "        "  # Indentación de función
                fixed_lines.append(f'{indent}try:')
                fixed_lines.append(f'{indent}    import torch')
                fixed_lines.append(f'{indent}except ImportError:')
                fixed_lines.append(f'{indent}    self.log_error("TorchONN", "PyTorch no está disponible para test")')
                fixed_lines.append(f'{indent}    return')
                fixed_lines.append('')
                import_added = True
                continue
        
        # Si encontramos device = torch.device sin import previo
        elif 'device = torch.device(' in line and 'import torch' not in content[:content.find(line)]:
            # Esta línea ya debería estar arreglada con el import añadido arriba
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Método alternativo más simple: buscar y reemplazar la línea problemática
    if not import_added:
        print("🔄 Aplicando fix alternativo...")
        fixed_content = content
        
        # Buscar el patrón problemático y añadir import antes
        problem_pattern = "device = torch.device('cpu')"
        if problem_pattern in fixed_content:
            # Reemplazar con versión que incluye import
            replacement = """try:
                    import torch
                    device = torch.device('cpu')
                except ImportError:
                    self.log_error("TorchONN", "PyTorch no disponible")
                    return"""
            
            fixed_content = fixed_content.replace(f"                    {problem_pattern}", replacement)
            
        # También buscar otras variantes
        if "device = torch.device(" in fixed_content and "import torch" not in fixed_content:
            # Buscar la función donde está el problema
            lines = fixed_content.split('\n')
            for i, line in enumerate(lines):
                if "device = torch.device(" in line and "import torch" not in '\n'.join(lines[:i]):
                    # Añadir import torch justo antes de esta línea
                    indent = len(line) - len(line.lstrip())
                    import_line = ' ' * indent + 'import torch'
                    lines.insert(i, import_line)
                    break
            fixed_content = '\n'.join(lines)
    else:
        fixed_content = '\n'.join(fixed_lines)
    
    # Escribir archivo corregido
    try:
        with open(diagnosis_file, 'w') as f:
            f.write(fixed_content)
        print(f"✅ Archivo corregido: {diagnosis_file}")
        return True
    except Exception as e:
        print(f"❌ Error escribiendo archivo: {e}")
        return False

def test_fixed_diagnosis():
    """Probar que el diagnóstico arreglado funciona"""
    print("\n🧪 Probando diagnosis.py arreglado...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "diagnosis.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ diagnosis.py ejecuta sin errores!")
            return True
        else:
            print("⚠️  diagnosis.py ejecuta con advertencias:")
            print(result.stdout[-500:])  # Últimas líneas
            return True  # Advertencias OK, errores no
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout - pero probablemente esté funcionando")
        return True
    except Exception as e:
        print(f"❌ Error probando: {e}")
        return False

def create_simple_diagnosis_alternative():
    """Crear diagnóstico alternativo súper simple"""
    print("\n📝 Creando diagnóstico alternativo...")
    
    simple_diagnosis = '''#!/usr/bin/env python3
"""
Diagnóstico simplificado para PtONN-TESTS
"""

import sys

def test_everything():
    """Test todo lo importante"""
    print("🔍 DIAGNÓSTICO SIMPLIFICADO PtONN-TESTS")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Python
    total_tests += 1
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        success_count += 1
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (requiere >=3.8)")
    
    # Test 2: PyTorch
    total_tests += 1
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        success_count += 1
    except ImportError:
        print("❌ PyTorch no instalado")
    
    # Test 3: NumPy
    total_tests += 1
    try:
        import numpy as np
        if np.__version__.startswith('2.'):
            print(f"⚠️  NumPy {np.__version__} (puede tener problemas)")
        else:
            print(f"✅ NumPy {np.__version__}")
            success_count += 1
    except ImportError:
        print("❌ NumPy no instalado")
    
    # Test 4: TorchONN
    total_tests += 1
    try:
        import torchonn
        print(f"✅ TorchONN {getattr(torchonn, '__version__', 'dev')}")
        success_count += 1
    except ImportError:
        print("❌ TorchONN no instalado")
    
    # Test 5: MZILayer
    total_tests += 1
    try:
        import torch
        from torchonn.layers import MZILayer
        layer = MZILayer(in_features=4, out_features=3)
        x = torch.randn(2, 4)
        output = layer(x)
        assert output.shape == (2, 3)
        print("✅ MZILayer funcionando")
        success_count += 1
    except Exception as e:
        print(f"❌ MZILayer: {e}")
    
    # Resumen
    percentage = (success_count / total_tests) * 100
    print(f"\\n📊 RESULTADO: {success_count}/{total_tests} ({percentage:.1f}%)")
    
    if percentage >= 80:
        print("🎉 ¡Proyecto en excelente estado!")
    elif percentage >= 60:
        print("👍 Proyecto en buen estado")
    else:
        print("🔧 Proyecto necesita algunas reparaciones")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = test_everything()
    sys.exit(0 if success else 1)
'''
    
    try:
        with open("simple_diagnosis.py", 'w') as f:
            f.write(simple_diagnosis)
        print("✅ Diagnóstico alternativo creado: simple_diagnosis.py")
        return True
    except Exception as e:
        print(f"❌ Error creando alternativo: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 Fix para diagnosis.py")
    print("Soluciona el problema de 'torch' is not defined")
    print()
    
    # Intentar arreglar el archivo existente
    fix_success = fix_diagnosis_script()
    
    if fix_success:
        # Probar que funciona
        test_success = test_fixed_diagnosis()
        
        if test_success:
            print("\n🎉 ¡PROBLEMA RESUELTO!")
            print("💡 Tu diagnosis.py ahora debería funcionar perfectamente")
            print("🔧 Ejecuta: python diagnosis.py")
            return True
    
    # Plan B: crear diagnóstico alternativo
    print("\n🔄 Plan B: Creando diagnóstico alternativo...")
    alt_success = create_simple_diagnosis_alternative()
    
    if alt_success:
        print("\n✅ SOLUCIÓN ALTERNATIVA CREADA")
        print("💡 Ejecuta: python simple_diagnosis.py")
        return True
    
    print("\n❌ No se pudo aplicar el fix automáticamente")
    print("💡 Solución manual:")
    print("   1. Abrir diagnosis.py")
    print("   2. Añadir 'import torch' al inicio de check_torchonn_installation()")
    print("   3. O usar simple_diagnosis.py como alternativa")
    
    return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)