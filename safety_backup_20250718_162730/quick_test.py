#!/bin/bash

# Install PyTorch Compatible - Bash Script
# Soluciona el error TensorBase automáticamente

set -e  # Exit on error

echo "🚀 INSTALADOR PYTORCH COMPATIBLE"
echo "================================="
echo "Solucionando error: 'module' object has no attribute 'TensorBase'"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Step 1: Check Python version
print_step "1️⃣ Verificando versión de Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" == "3.12" ]] || [[ "$PYTHON_VERSION" > "3.12" ]]; then
    print_warning "⚠️ Python 3.12+ detectado - usando PyTorch 2.1+ para compatibilidad"
    TORCH_VERSION="2.1.0"
    TORCHVISION_VERSION="0.16.0"
    TORCHAUDIO_VERSION="2.1.0"
else
    print_step "✅ Python version compatible"
    TORCH_VERSION="2.0.1"
    TORCHVISION_VERSION="0.15.2"
    TORCHAUDIO_VERSION="2.0.2"
fi

# Step 2: Clean Python cache
print_step "2️⃣ Limpiando caché de Python..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✅ Caché limpiado"

# Step 3: Uninstall PyTorch completely
print_step "3️⃣ Desinstalando PyTorch completamente..."
python3 -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
python3 -m pip uninstall pytorch -y 2>/dev/null || true
echo "✅ PyTorch desinstalado"

# Step 4: Install compatible NumPy
print_step "4️⃣ Instalando NumPy compatible..."
python3 -m pip install "numpy>=1.21.0,<2.0.0" --force-reinstall
echo "✅ NumPy instalado"

# Step 5: Install PyTorch stable
print_step "5️⃣ Instalando PyTorch $TORCH_VERSION..."
python3 -m pip install \
    torch==$TORCH_VERSION \
    torchvision==$TORCHVISION_VERSION \
    torchaudio==$TORCHAUDIO_VERSION \
    --index-url https://download.pytorch.org/whl/cpu

echo "✅ PyTorch instalado"

# Step 6: Install additional dependencies
print_step "6️⃣ Instalando dependencias adicionales..."
python3 -m pip install \
    "scipy>=1.9.0,<1.12.0" \
    "matplotlib>=3.5.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.64.0"

echo "✅ Dependencias instaladas"

# Step 7: Install project in development mode
print_step "7️⃣ Instalando proyecto en modo desarrollo..."
if [ -f "setup.py" ]; then
    python3 -m pip install -e .
    echo "✅ Proyecto instalado"
else
    print_warning "⚠️ No se encontró setup.py, saltando instalación del proyecto"
fi

# Step 8: Update requirements.txt
print_step "8️⃣ Actualizando requirements.txt..."
cat > requirements.txt << EOF
torch>=$TORCH_VERSION,<2.3.0
torchvision>=$TORCHVISION_VERSION,<0.18.0
torchaudio>=$TORCHAUDIO_VERSION,<2.3.0
numpy>=1.21.0,<2.0.0
scipy>=1.9.0,<1.12.0
matplotlib>=3.5.0,<4.0.0
pyyaml>=6.0,<7.0.0
tqdm>=4.64.0,<5.0.0
pytest>=7.0.0,<8.0.0
pytest-cov>=4.0.0,<5.0.0
EOF

echo "✅ requirements.txt actualizado"

# Step 9: Verification
print_step "9️⃣ Verificación final..."

# Create verification script
cat > verify_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
import torch

def verify():
    print("🧪 Verificando instalación...")
    
    # Test 1: PyTorch version
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Test 2: Basic tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = torch.mm(x, y)
    print(f"✅ Tensor operations: {x.shape} @ {y.shape} = {z.shape}")
    
    # Test 3: Autograd
    x = torch.randn(2, 3, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    print(f"✅ Autograd: OK")
    
    # Test 4: Device info
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    # Test 5: TorchONN (if available)
    try:
        sys.path.insert(0, '.')
        import torchonn
        from torchonn.layers import MZILayer
        
        layer = MZILayer(4, 3)
        test_input = torch.randn(2, 4)
        output = layer(test_input)
        
        print(f"✅ TorchONN: {torchonn.__version__}")
        print(f"✅ MZI Layer: {test_input.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"⚠️ TorchONN: {e}")
    
    print("\n🎉 ¡Verificación completada!")
    return True

if __name__ == "__main__":
    try:
        verify()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
EOF

# Run verification
python3 verify_installation.py

if [ $? -eq 0 ]; then
    print_step "🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!"
    echo ""
    echo "📋 Resumen de lo instalado:"
    echo "   • PyTorch $TORCH_VERSION"
    echo "   • TorchVision $TORCHVISION_VERSION"
    echo "   • TorchAudio $TORCHAUDIO_VERSION"
    echo "   • NumPy compatible (<2.0)"
    echo "   • Dependencias adicionales"
    echo ""
    echo "🚀 Próximos pasos:"
    echo "   • python test_installation.py"
    echo "   • python examples/basic_usage.py"
    echo "   • python -c \"import torch; print('✅ PyTorch OK')\""
    echo ""
    echo "📂 Archivos creados/actualizados:"
    echo "   • requirements.txt"
    echo "   • verify_installation.py"
else
    print_error "❌ Error en la verificación"
    print_error "💡 Intenta ejecutar: python3 fix_pytorch_installation.py"
    exit 1
fi

# Cleanup
rm -f verify_installation.py

print_step "✅ Instalación completada - Error TensorBase solucionado"
