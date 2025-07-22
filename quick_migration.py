#!/usr/bin/env python3
"""
🔧 PtONN-TESTS: Corrector de Imports Post-Migración

Script de corrección rápida para arreglar el problema de imports faltantes
en los módulos migrados de componentes fotónicos.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

class ImportsFixer:
    """Corrector de imports para módulos migrados."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        
        # Archivos a corregir
        self.files_to_fix = [
            "torchonn/layers/microring.py",
            "torchonn/layers/couplers.py", 
            "torchonn/layers/detectors.py",
            "torchonn/components/memory.py",
            "torchonn/components/wdm.py"
        ]
        
        # Imports estándar que deben estar en todos los archivos
        self.standard_imports = [
            "import torch",
            "import torch.nn as nn",
            "import numpy as np",
            "from typing import List, Tuple, Optional, Dict, Union",
            "import math",
            "import warnings"
        ]
        
        print(f"🔧 Corrector de imports inicializado en: {self.repo_root}")
    
    def get_correct_header(self, module_name: str) -> str:
        """Obtener header correcto para cada módulo."""
        
        headers = {
            "microring.py": '''"""
Microring Photonic Components for PtONN-TESTS

Implementation of microring resonators and related components
for photonic neural network simulation.
"""''',
            
            "couplers.py": '''"""
Coupler Components for PtONN-TESTS

Implementation of directional couplers and beam splitters
for photonic neural network simulation.
"""''',
            
            "detectors.py": '''"""
Detector Components for PtONN-TESTS  

Implementation of photodetectors and optical-to-electrical
conversion components for photonic neural networks.
"""''',
            
            "memory.py": '''"""
Memory Components for PtONN-TESTS

Implementation of phase change materials and other
non-volatile memory components for photonic computing.
"""''',
            
            "wdm.py": '''"""
WDM Components for PtONN-TESTS

Implementation of wavelength division multiplexing
and related systems for photonic neural networks.
"""'''
        }
        
        return headers.get(module_name, '"""Photonic Components for PtONN-TESTS"""')
    
    def fix_file_imports(self, file_path: Path) -> bool:
        """Corregir imports de un archivo específico."""
        
        if not file_path.exists():
            print(f"  ⚠️  Archivo no existe: {file_path}")
            return False
        
        try:
            # Leer contenido actual
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Encontrar donde empiezan las clases (después de imports/header)
            class_start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('class '):
                    class_start_idx = i
                    break
            
            if class_start_idx == 0:
                print(f"  ⚠️  No se encontraron clases en: {file_path}")
                return False
            
            # Obtener solo el código de las clases
            class_content = '\n'.join(lines[class_start_idx:])
            
            # Crear nuevo contenido con header e imports correctos
            module_name = file_path.name
            new_content = []
            
            # Header
            new_content.append(self.get_correct_header(module_name))
            new_content.append('')
            
            # Imports estándar
            new_content.extend(self.standard_imports)
            new_content.append('')
            
            # Clases (contenido existente)
            new_content.append(class_content)
            
            # Escribir archivo corregido
            corrected_content = '\n'.join(new_content)
            file_path.write_text(corrected_content, encoding='utf-8')
            
            print(f"  ✅ Corregido: {file_path.name}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error corrigiendo {file_path}: {e}")
            return False
    
    def verify_fix(self, file_path: Path) -> Dict[str, bool]:
        """Verificar que la corrección funcionó."""
        
        results = {
            "file_exists": False,
            "has_torch_import": False,
            "has_nn_import": False,
            "has_numpy_import": False,
            "has_typing_import": False,
            "has_classes": False
        }
        
        if not file_path.exists():
            return results
        
        results["file_exists"] = True
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            results["has_torch_import"] = "import torch" in content
            results["has_nn_import"] = "torch.nn as nn" in content
            results["has_numpy_import"] = "import numpy as np" in content
            results["has_typing_import"] = "from typing import" in content
            results["has_classes"] = "class " in content
            
        except Exception:
            pass
        
        return results
    
    def create_backup(self) -> Path:
        """Crear backup antes de corregir."""
        from datetime import datetime
        
        backup_dir = self.repo_root / f"imports_fix_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)
        
        for file_rel in self.files_to_fix:
            file_path = self.repo_root / file_rel
            if file_path.exists():
                backup_path = backup_dir / file_rel
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
        
        print(f"📁 Backup creado en: {backup_dir}")
        return backup_dir
    
    def run_fix(self) -> bool:
        """Ejecutar corrección completa."""
        print("🚀 INICIANDO CORRECCIÓN DE IMPORTS")
        print("=" * 60)
        
        # Crear backup
        backup_dir = self.create_backup()
        
        # Corregir cada archivo
        print("\n🔧 Corrigiendo archivos:")
        success_count = 0
        
        for file_rel in self.files_to_fix:
            file_path = self.repo_root / file_rel
            success = self.fix_file_imports(file_path)
            if success:
                success_count += 1
        
        # Verificar correcciones
        print(f"\n🔍 Verificando correcciones:")
        all_good = True
        
        for file_rel in self.files_to_fix:
            file_path = self.repo_root / file_rel
            results = self.verify_fix(file_path)
            
            if all(results.values()):
                print(f"  ✅ {file_path.name} - todos los imports correctos")
            else:
                print(f"  ❌ {file_path.name} - faltan imports:")
                for check, passed in results.items():
                    if not passed:
                        print(f"     - {check}")
                all_good = False
        
        # Resumen
        print("\n" + "=" * 60)
        if all_good and success_count == len(self.files_to_fix):
            print("✅ CORRECCIÓN EXITOSA!")
            print(f"   📁 Archivos corregidos: {success_count}/{len(self.files_to_fix)}")
            print(f"   📁 Backup disponible en: {backup_dir}")
            print("\n🎯 Próximo paso:")
            print("   python validate_migration.py")
            return True
        else:
            print("❌ CORRECCIÓN INCOMPLETA")
            print(f"   📁 Archivos corregidos: {success_count}/{len(self.files_to_fix)}")
            print(f"   📁 Backup disponible en: {backup_dir}")
            return False

def main():
    """Función principal."""
    print("🔧 PtONN-TESTS: Corrector de Imports Post-Migración")
    print("🎯 Arreglando problema 'name nn is not defined'")
    print()
    
    # Detectar directorio del repositorio
    repo_root = Path.cwd()
    
    # Verificar que estamos en el directorio correcto
    if not (repo_root / "torchonn").exists():
        print("❌ Error: No se encontró directorio torchonn/")
        print("   Por favor ejecutar desde el directorio raíz del repositorio")
        sys.exit(1)
    
    # Crear y ejecutar corrector
    fixer = ImportsFixer(repo_root)
    success = fixer.run_fix()
    
    if success:
        print("\n🎉 ¡Imports corregidos exitosamente!")
        print("💡 Ahora ejecuta: python validate_migration.py")
        sys.exit(0)
    else:
        print("\n❌ Corrección falló")
        sys.exit(1)

if __name__ == "__main__":
    main()