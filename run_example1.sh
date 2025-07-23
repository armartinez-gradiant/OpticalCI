#!/bin/bash
# 🚀 EJECUTOR DE EXAMPLE1.PY
# Soluciona automáticamente problemas de paths

echo "🚀 Ejecutando Example1.py con paths corregidos..."

# Método 1: Añadir PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Método 2: Ejecutar
python examples/Example1.py

# Si falla, intentar método alternativo
if [ $? -ne 0 ]; then
    echo "⚠️ Método 1 falló, intentando método 2..."
    cd examples/
    export PYTHONPATH=$PYTHONPATH:..
    python Example1.py
    cd ..
fi
