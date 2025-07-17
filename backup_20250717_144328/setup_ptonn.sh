#!/bin/bash

# Script de instalación para PtONN-TESTS
# Instala pytorch-onn y todas sus dependencias correctamente

set -e  # Salir en caso de error

echo "🚀 Configurando entorno para PtONN-TESTS..."

# Verificar Python
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "📋 Python version detectada: $PYTHON_VERSION"

# Verificar compatibilidad con PyTorch
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "⚠️  ADVERTENCIA: Python 3.13 no es totalmente compatible con PyTorch"
    echo "   Se recomienda usar Python 3.11 o 3.10"
    exit 1
fi

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "🔧 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Instalar PyTorch primero
echo "🔥 Instalando PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias básicas
echo "📦 Instalando dependencias básicas..."
pip install numpy scipy matplotlib pyyaml tqdm

# Instalar pyutils (dependencia opcional de pytorch-onn)
echo "🔧 Instalando pyutils (opcional)..."
pip install git+https://github.com/JeremieMelo/pyutility.git || echo "⚠️  pyutils no instalado, no es crítico"

# Instalar pytorch-onn
echo "🔬 Instalando pytorch-onn..."
echo "   Intentando instalación desde GitHub..."
if pip install git+https://github.com/JeremieMelo/pytorch-onn.git; then
    echo "✅ pytorch-onn instalado desde GitHub"
else
    echo "⚠️  Instalación desde GitHub falló, intentando PyPI..."
    if pip install pytorch-onn; then
        echo "✅ pytorch-onn instalado desde PyPI"
    else
        echo "⚠️  Instalación desde PyPI falló, clonando repositorio..."
        git clone https://github.com/JeremieMelo/pytorch-onn.git temp_pytorch_onn
        cd temp_pytorch_onn
        pip install -e .
        cd ..
        rm -rf temp_pytorch_onn
        echo "✅ pytorch-onn instalado desde código fuente"
    fi
fi

# Instalar dependencias de testing
echo "🧪 Instalando dependencias de testing..."
pip install pytest pytest-cov

# Instalar el proyecto si existe setup.py
if [ -f "setup.py" ]; then
    echo "🔧 Instalando proyecto local..."
    pip install -e .
fi

# Verificar instalación
echo "✅ Verificando instalación..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchonn; print('torchonn module imported successfully')" || {
    echo "❌ Error: torchonn no se pudo importar"
    echo "   Intentando diagnóstico..."
    python -c "import sys; print('Python path:'); [print(p) for p in sys.path]"
    pip show pytorch-onn || echo "pytorch-onn no está instalado"
    exit 1
}

# Verificar componentes específicos
echo "🔍 Verificando componentes específicos..."
python -c "from torchonn.layers import MZILayer, MZIBlockLinear; print('MZI layers imported successfully')" || {
    echo "❌ Error: No se pudieron importar MZI layers"
    echo "   Esto puede indicar una versión incompatible de pytorch-onn"
}

# Ejecutar un test básico
echo "🧪 Ejecutando test básico..."
python -c "
import torch
import torchonn
print('Basic test passed!')
"

echo "🎉 ¡Instalación completada!"
echo "📝 Para activar el entorno virtual: source venv/bin/activate"
echo "🧪 Para ejecutar tests: pytest unittest/"
echo "🧪 Para verificar instalación: python test_installation.py"
echo "🚀 Para desactivar el entorno: deactivate"

# Mostrar información del entorno
echo "📊 Información del entorno:"
pip list | grep -E "(torch|onn|numpy|scipy|pyutils)"