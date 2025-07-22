#!/usr/bin/env python3
"""
Version Reset Script - PtONN-TESTS

Script para resetear la versión del proyecto a 1.0.0 y hacer commit
de todos los archivos con la nueva versión.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

def touch_all_files(repo_root):
    """Tocar todos los archivos para forzar que Git los vea como modificados."""
    print("Touching all files to force Git to see them as modified...")
    
    # Obtener todos los archivos trackeados por Git
    result = subprocess.run(['git', 'ls-files'], 
                          capture_output=True, text=True, cwd=repo_root)
    
    if result.returncode != 0:
        print("Error: Failed to get Git tracked files")
        return False
    
    files = result.stdout.strip().split('\n')
    
    for file_path in files:
        full_path = repo_root / file_path
        if full_path.exists():
            # Touch the file (update modification time)
            full_path.touch()
    
    print(f"Touched {len(files)} files")
    return True

def update_version_in_files(repo_root):
    """Actualizar números de versión en archivos clave."""
    print("Updating version numbers in key files...")
    
    version_files = {
        'torchonn/__init__.py': r'__version__ = ".*"',
        'setup.py': r'version=".*"',
        'pyproject.toml': r'version = ".*"'
    }
    
    for file_path, pattern in version_files.items():
        full_path = repo_root / file_path
        if full_path.exists():
            try:
                content = full_path.read_text(encoding='utf-8')
                
                # Replace version
                import re
                if 'pyproject.toml' in file_path:
                    new_content = re.sub(pattern, 'version = "1.0.0"', content)
                elif 'setup.py' in file_path:
                    new_content = re.sub(pattern, 'version="1.0.0"', content)
                else:  # __init__.py
                    new_content = re.sub(pattern, '__version__ = "1.0.0"', content)
                
                if new_content != content:
                    full_path.write_text(new_content, encoding='utf-8')
                    print(f"  Updated: {file_path}")
                
            except Exception as e:
                print(f"  Error updating {file_path}: {e}")

def main():
    """Función principal."""
    print("PtONN-TESTS: Version Reset to 1.0.0")
    print("=" * 50)
    
    repo_root = Path.cwd()
    
    # Verificar que estamos en un repo Git
    if not (repo_root / '.git').exists():
        print("Error: Not in a Git repository")
        return False
    
    print("This will:")
    print("1. Update version numbers to 1.0.0 in key files")
    print("2. Touch all files to mark them as modified") 
    print("3. Stage all changes")
    print("4. Create a commit with version 1.0.0")
    print("5. Create a Git tag v1.0.0")
    print()
    
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled")
        return False
    
    try:
        # 1. Update version numbers
        update_version_in_files(repo_root)
        
        # 2. Touch all files  
        touch_all_files(repo_root)
        
        # 3. Stage all changes
        print("\\nStaging all changes...")
        result = subprocess.run(['git', 'add', '-A'], cwd=repo_root)
        if result.returncode != 0:
            print("Error: Failed to stage changes")
            return False
        
        # 4. Create commit
        print("Creating commit...")
        commit_message = "Release v1.0.0\\n\\nInitial stable release of PtONN-TESTS with:\\n- Modern PyTorch compatibility\\n- Complete photonic component library\\n- Comprehensive testing suite\\n- Updated authorship and licensing"
        
        result = subprocess.run(['git', 'commit', '-m', commit_message], cwd=repo_root)
        if result.returncode != 0:
            print("Error: Failed to create commit")
            return False
        
        # 5. Create tag
        print("Creating Git tag v1.0.0...")
        result = subprocess.run(['git', 'tag', '-a', 'v1.0.0', '-m', 'Version 1.0.0 - Initial stable release'], cwd=repo_root)
        if result.returncode != 0:
            print("Warning: Failed to create tag (may already exist)")
        
        print("\\nSuccess! All files are now at version 1.0.0")
        print("\\nNext steps:")
        print("  git push origin main")
        print("  git push origin --tags")
        
        return True
        
    except Exception as e:
        print(f"\\nError: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)