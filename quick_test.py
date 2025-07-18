#!/usr/bin/env python3
"""
Diagnóstico Avanzado PtONN-TESTS
================================

Script mejorado para detectar y diagnosticar problemas en el repositorio.
"""

import sys
import os
from pathlib import Path
import subprocess
import importlib.util
import ast
import traceback
from typing import Dict, List, Tuple, Optional

# Add current directory to path
current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))

class PtONNDiagnostic:
    def __init__(self):
        self.repo_path = Path.cwd()
        self.issues = []
        self.warnings = []
        self.successes = []
        self.step_count = 0
        
    def log_step(self, title: str):
        """Log a diagnostic step"""
        self.step_count += 1
        print(f"\n{self.step_count}️⃣ {title}")
        print("-" * 50)
        
    def log_result(self, message: str, status: str = "info"):
        """Log a result with status"""
        if status == "success":
            print(f"  ✅ {message}")
            self.successes.append(message)
        elif status == "warning":
            print(f"  ⚠️  {message}")
            self.warnings.append(message)
        elif status == "error":
            print(f"  ❌ {message}")
            self.issues.append(message)
        else:
            print(f"  ℹ️  {message}")

    def check_file_structure(self):
        """Check basic file structure"""
        self.log_step("VERIFICANDO ESTRUCTURA DE ARCHIVOS")
        
        # Essential files and directories
        essential_paths = [
            "torchonn/__init__.py",
            "torchonn/layers/__init__.py",
            "torchonn/layers/mzi_layer.py",
            "torchonn/layers/mzi_block_linear.py",
            "torchonn/models/__init__.py",
            "torchonn/models/base_model.py",
            "torchonn/devices/__init__.py",
            "torchonn/ops/__init__.py",
            "torchonn/utils/__init__.py",
        ]
        
        for path_str in essential_paths:
            path = self.repo_path / path_str
            if path.exists():
                self.log_result(f"Found: {path_str}", "success")
            else:
                self.log_result(f"Missing: {path_str}", "error")
        
        # Check for common issues
        pycache_dirs = list(self.repo_path.rglob("__pycache__"))
        if pycache_dirs:
            self.log_result(f"Found {len(pycache_dirs)} __pycache__ directories (should clean)", "warning")
        
        pyc_files = list(self.repo_path.rglob("*.pyc"))
        if pyc_files:
            self.log_result(f"Found {len(pyc_files)} .pyc files (should clean)", "warning")

    def check_file_syntax(self):
        """Check Python file syntax"""
        self.log_step("VERIFICANDO SINTAXIS DE ARCHIVOS PYTHON")
        
        python_files = list(self.repo_path.rglob("*.py"))
        syntax_errors = []
        empty_files = []
        
        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    empty_files.append(py_file)
                    continue
                    
                # Try to parse as AST
                ast.parse(content)
                
            except SyntaxError as e:
                syntax_errors.append((py_file, str(e)))
            except UnicodeDecodeError as e:
                syntax_errors.append((py_file, f"Unicode error: {e}"))
            except Exception as e:
                syntax_errors.append((py_file, f"Unknown error: {e}"))
        
        if syntax_errors:
            self.log_result(f"Found {len(syntax_errors)} files with syntax errors:", "error")
            for file_path, error in syntax_errors[:5]:  # Show first 5
                self.log_result(f"  {file_path.relative_to(self.repo_path)}: {error}", "error")
            if len(syntax_errors) > 5:
                self.log_result(f"  ... and {len(syntax_errors) - 5} more", "error")
        else:
            self.log_result("All Python files have valid syntax", "success")
        
        if empty_files:
            self.log_result(f"Found {len(empty_files)} empty Python files", "warning")
            for empty in empty_files[:3]:
                self.log_result(f"  {empty.relative_to(self.repo_path)}", "warning")

    def check_imports_detailed(self):
        """Detailed import checking"""
        self.log_step("VERIFICANDO IMPORTS DETALLADOS")
        
        # Test basic imports
        import_tests = [
            ("torch", "PyTorch core"),
            ("numpy", "NumPy"),
            ("torchonn", "TorchONN main package"),
            ("torchonn.layers", "TorchONN layers module"),
            ("torchonn.models", "TorchONN models module"),
            ("torchonn.devices", "TorchONN devices module"),
            ("torchonn.ops", "TorchONN operations module"),
            ("torchonn.utils", "TorchONN utilities module"),
        ]
        
        for module_name, description in import_tests:
            try:
                # Try to find and load the module
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    self.log_result(f"{description}: Module not found", "error")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get additional info if available
                version = getattr(module, '__version__', 'unknown')
                self.log_result(f"{description}: OK (v{version})", "success")
                
            except Exception as e:
                self.log_result(f"{description}: Import failed - {str(e)[:100]}", "error")

    def check_class_imports(self):
        """Check specific class imports"""
        self.log_step("VERIFICANDO IMPORTS DE CLASES ESPECÍFICAS")
        
        class_tests = [
            ("torchonn.layers.MZILayer", "MZI Layer class"),
            ("torchonn.layers.MZIBlockLinear", "MZI Block Linear class"),
            ("torchonn.models.ONNBaseModel", "ONN Base Model class"),
        ]
        
        for import_path, description in class_tests:
            try:
                module_path, class_name = import_path.rsplit('.', 1)
                spec = importlib.util.find_spec(module_path)
                if spec is None:
                    self.log_result(f"{description}: Module {module_path} not found", "error")
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    self.log_result(f"{description}: OK ({cls})", "success")
                else:
                    self.log_result(f"{description}: Class {class_name} not found in module", "error")
                    # Show available attributes
                    available = [attr for attr in dir(module) if not attr.startswith('_')]
                    self.log_result(f"  Available: {available[:5]}", "info")
                
            except Exception as e:
                self.log_result(f"{description}: Failed - {str(e)[:100]}", "error")

    def test_basic_functionality(self):
        """Test basic functionality"""
        self.log_step("PROBANDO FUNCIONALIDAD BÁSICA")
        
        try:
            # Test torch
            import torch
            x = torch.randn(2, 4)
            self.log_result(f"PyTorch tensor creation: {x.shape}", "success")
            
            # Test torchonn imports
            import torchonn
            self.log_result(f"TorchONN version: {getattr(torchonn, '__version__', 'unknown')}", "success")
            
            # Test specific classes
            from torchonn.layers import MZILayer, MZIBlockLinear
            
            # Test MZI Layer
            layer = MZILayer(4, 3)
            output = layer(x)
            self.log_result(f"MZI Layer: {x.shape} -> {output.shape}", "success")
            
            # Test MZI Block
            block = MZIBlockLinear(4, 3, mode="usv")
            output2 = block(x)
            self.log_result(f"MZI Block: {x.shape} -> {output2.shape}", "success")
            
            # Test gradient computation
            output.sum().backward()
            self.log_result("Gradient computation: OK", "success")
            
            # Test model
            from torchonn.models import ONNBaseModel
            
            class TestModel(ONNBaseModel):
                def __init__(self):
                    super().__init__()
                    self.layer = MZILayer(4, 2)
                
                def forward(self, x):
                    return self.layer(x)
            
            model = TestModel()
            model_output = model(x)
            self.log_result(f"Complete model: {x.shape} -> {model_output.shape}", "success")
            
        except Exception as e:
            self.log_result(f"Functionality test failed: {str(e)}", "error")
            traceback.print_exc()

    def check_stub_files(self):
        """Check for stub files that need implementation"""
        self.log_step("VERIFICANDO ARCHIVOS STUB (TODO)")
        
        stub_files = []
        todo_files = []
        
        for py_file in self.repo_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "TODO: Implementar" in content:
                    stub_files.append(py_file)
                
                if "# TODO" in content or "TODO:" in content:
                    todo_files.append(py_file)
                    
            except Exception:
                continue
        
        if stub_files:
            self.log_result(f"Found {len(stub_files)} stub files needing implementation:", "warning")
            for stub in stub_files[:5]:
                self.log_result(f"  {stub.relative_to(self.repo_path)}", "warning")
        
        if todo_files:
            self.log_result(f"Found {len(todo_files)} files with TODO comments:", "info")

    def check_circular_imports(self):
        """Check for potential circular imports"""
        self.log_step("VERIFICANDO IMPORTS CIRCULARES")
        
        # This is a simplified check - in reality, circular imports are complex to detect
        import_graph = {}
        
        for py_file in self.repo_path.rglob("*.py"):
            if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple regex to find imports
                imports = []
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('from torchonn') or line.startswith('import torchonn'):
                        imports.append(line)
                
                if imports:
                    rel_path = str(py_file.relative_to(self.repo_path))
                    import_graph[rel_path] = imports
                    
            except Exception:
                continue
        
        if import_graph:
            self.log_result(f"Analyzed {len(import_graph)} files for import patterns", "info")
            # Look for potential issues
            cross_imports = 0
            for file_path, imports in import_graph.items():
                for imp in imports:
                    if "components" in imp and "components" not in file_path:
                        cross_imports += 1
            
            if cross_imports > 0:
                self.log_result(f"Found {cross_imports} potential cross-module imports", "warning")
            else:
                self.log_result("No obvious circular import patterns detected", "success")

    def check_dependencies(self):
        """Check dependencies and versions"""
        self.log_step("VERIFICANDO DEPENDENCIAS Y VERSIONES")
        
        try:
            # Check requirements.txt
            req_file = self.repo_path / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'r') as f:
                    requirements = f.read().strip().split('\n')
                self.log_result(f"Found requirements.txt with {len(requirements)} dependencies", "success")
            else:
                self.log_result("No requirements.txt found", "warning")
            
            # Check key dependencies
            deps_to_check = [
                ("torch", "2.0.0"),
                ("numpy", "1.19.0"),
                ("scipy", None),
                ("matplotlib", None),
            ]
            
            for dep_name, min_version in deps_to_check:
                try:
                    dep = importlib.import_module(dep_name)
                    version = getattr(dep, '__version__', 'unknown')
                    self.log_result(f"{dep_name}: {version}", "success")
                    
                    if min_version and hasattr(dep, '__version__'):
                        # Simple version comparison
                        if version < min_version:
                            self.log_result(f"  Version {version} may be too old (need >= {min_version})", "warning")
                        
                except ImportError:
                    self.log_result(f"{dep_name}: Not installed", "error")
                    
        except Exception as e:
            self.log_result(f"Dependency check failed: {e}", "error")

    def suggest_fixes(self):
        """Suggest fixes based on found issues"""
        self.log_step("SUGERENCIAS DE CORRECCIÓN")
        
        if not self.issues:
            self.log_result("No critical issues found! 🎉", "success")
            return
        
        fixes = []
        
        # Check for common issues and suggest fixes
        for issue in self.issues:
            if "syntax error" in issue.lower():
                fixes.append("🔧 Fix syntax errors in Python files")
            elif "module not found" in issue.lower():
                fixes.append("🔧 Check Python path and module structure")
            elif "import failed" in issue.lower():
                fixes.append("🔧 Verify all imports and dependencies")
            elif "missing" in issue.lower():
                fixes.append("🔧 Create missing files and directories")
        
        # Add general fixes
        if self.warnings:
            fixes.append("🧹 Clean up stub files and implement missing functionality")
            fixes.append("🧹 Remove __pycache__ directories and .pyc files")
        
        fixes.append("📦 Ensure all dependencies are installed: pip install -r requirements.txt")
        fixes.append("🔧 Run: pip install -e . to install package in development mode")
        
        unique_fixes = list(set(fixes))
        for fix in unique_fixes:
            self.log_result(fix, "info")

    def run_comprehensive_diagnostic(self):
        """Run all diagnostic checks"""
        print("🔍 DIAGNÓSTICO COMPLETO PtONN-TESTS")
        print("=" * 60)
        
        self.check_file_structure()
        self.check_file_syntax()
        self.check_imports_detailed()
        self.check_class_imports()
        self.test_basic_functionality()
        self.check_stub_files()
        self.check_circular_imports()
        self.check_dependencies()
        self.suggest_fixes()
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 RESUMEN DEL DIAGNÓSTICO")
        print(f"{'='*60}")
        print(f"✅ Éxitos: {len(self.successes)}")
        print(f"⚠️  Advertencias: {len(self.warnings)}")
        print(f"❌ Errores críticos: {len(self.issues)}")
        
        if self.issues:
            print(f"\n🚨 ERRORES CRÍTICOS ENCONTRADOS:")
            for issue in self.issues[:5]:  # Show first 5
                print(f"  • {issue}")
            if len(self.issues) > 5:
                print(f"  • ... y {len(self.issues) - 5} más")
        
        success_rate = len(self.successes) / (len(self.successes) + len(self.issues) + len(self.warnings)) * 100
        print(f"\n📈 Tasa de éxito: {success_rate:.1f}%")
        
        if self.issues:
            print("\n🔧 SIGUIENTE PASO: Revisar y corregir los errores críticos listados arriba")
            return False
        elif self.warnings:
            print("\n✅ Sistema funcional pero con advertencias menores")
            return True
        else:
            print("\n🎉 ¡PERFECTO! Todo funciona correctamente")
            return True

def main():
    try:
        diagnostic = PtONNDiagnostic()
        success = diagnostic.run_comprehensive_diagnostic()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Diagnóstico interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error durante diagnóstico: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
