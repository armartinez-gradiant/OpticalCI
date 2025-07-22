#!/usr/bin/env python3
"""
🔍 Diagnóstico rápido de imports en módulos migrados
"""

from pathlib import Path

def diagnose_file(file_path):
    """Diagnosticar un archivo específico."""
    if not file_path.exists():
        print(f"❌ {file_path} - NO EXISTE")
        return
    
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        print(f"\n📁 {file_path.name}:")
        print(f"   📏 Líneas: {len(lines)}")
        
        # Mostrar primeras 15 líneas para ver imports
        print("   🔍 Primeras líneas:")
        for i, line in enumerate(lines[:15]):
            print(f"   {i+1:2d}: {line}")
        
        # Buscar imports específicos
        imports_found = {
            "torch": "import torch" in content,
            "nn": "torch.nn as nn" in content or "import torch.nn as nn" in content,
            "numpy": "import numpy" in content,
            "typing": "from typing" in content,
            "warnings": "import warnings" in content or "warnings" in content
        }
        
        print("   📦 Imports encontrados:")
        for imp, found in imports_found.items():
            status = "✅" if found else "❌"
            print(f"      {status} {imp}")
        
        # Buscar clases
        classes = [line.strip() for line in lines if line.strip().startswith('class ')]
        print(f"   🏗️  Clases encontradas: {len(classes)}")
        for cls in classes:
            print(f"      - {cls}")
            
    except Exception as e:
        print(f"❌ Error leyendo {file_path}: {e}")

def main():
    repo_root = Path.cwd()
    
    files_to_check = [
        "torchonn/layers/microring.py",
        "torchonn/layers/couplers.py", 
        "torchonn/layers/detectors.py",
        "torchonn/components/memory.py",
        "torchonn/components/wdm.py"
    ]
    
    print("🔍 DIAGNÓSTICO RÁPIDO DE IMPORTS")
    print("=" * 50)
    
    for file_rel in files_to_check:
        file_path = repo_root / file_rel
        diagnose_file(file_path)
    
    print(f"\n💡 Ejecuta: python fix_migration_imports.py")

if __name__ == "__main__":
    main()