"""
Plantilla Básica para Scripts de ONN
====================================

Plantilla reutilizable para crear tus propios scripts de redes ópticas.
Ubicación: /workspaces/OpticalCI/scripts/templates/basic_onn_template.py

INSTRUCCIONES DE USO:
1. Copia este archivo a scripts/tu_proyecto/
2. Modifica las secciones marcadas con "# TODO"
3. Ejecuta tu script personalizado

Esta plantilla incluye:
- Setup básico de OpticalCI
- Estructura de entrenamiento estándar
- Extracción automática de phase shifters
- Guardado de resultados
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime

# ===================================================================
# SETUP BÁSICO - NO MODIFICAR
# ===================================================================

# Añadir path del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Importaciones de OpticalCI
try:
    from torchonn.layers import MZILayer, MZIBlockLinear
    from torchonn.utils.phase_shifter_extractor import PhaseShifterExtractor, quick_extract
    print("🌟 OpticalCI cargado exitosamente!")
except ImportError as e:
    print(f"❌ Error importando OpticalCI: {e}")
    sys.exit(1)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Usando device: {device}")

# ===================================================================
# CONFIGURACIÓN DEL PROYECTO - MODIFICA AQUÍ
# ===================================================================

# TODO: Configura los parámetros de tu proyecto
PROJECT_CONFIG = {
    "project_name": "mi_proyecto_onn",  # TODO: Cambia el nombre
    "description": "Descripción de tu proyecto",  # TODO: Añade descripción
    
    # Arquitectura de la red
    "architecture": {
        "input_size": 4,    # TODO: Cambia el tamaño de entrada
        "output_size": 4,   # TODO: Cambia el tamaño de salida
        "layer_type": "MZI"  # MZI o MZIBlock
    },
    
    # Parámetros de entrenamiento
    "training": {
        "learning_rate": 0.01,  # TODO: Ajusta learning rate
        "epochs": 100,          # TODO: Ajusta número de épocas
        "batch_size": 32,       # TODO: Ajusta batch size si es necesario
        "optimizer": "adam"     # adam, sgd, rmsprop
    },
    
    # Configuración de datos
    "data": {
        "n_samples": 1000,      # TODO: Número de muestras
        "noise_level": 0.1,     # TODO: Nivel de ruido
        "test_ratio": 0.2       # TODO: Proporción para test
    }
}

# ===================================================================
# FUNCIONES PARA DATOS - MODIFICA SEGÚN TU CASO
# ===================================================================

def generate_training_data():
    """
    Generar datos de entrenamiento.
    
    TODO: Modifica esta función según tu problema específico.
    Ejemplos:
    - Cargar datos desde archivo
    - Generar datos sintéticos
    - Usar funciones matemáticas específicas
    """
    config = PROJECT_CONFIG["data"]
    arch_config = PROJECT_CONFIG["architecture"]
    
    n_samples = config["n_samples"]
    input_size = arch_config["input_size"]
    output_size = arch_config["output_size"]
    
    # TODO: EJEMPLO - Reemplaza con tu lógica de generación de datos
    print("📊 Generando datos de ejemplo...")
    
    # Datos de entrada aleatorios
    X = torch.randn(n_samples, input_size, device=device) * 2.0
    
    # TODO: Define tu función objetivo aquí
    # Ejemplo: transformación lineal + no-linealidad
    W_target = torch.randn(output_size, input_size, device=device) * 0.5
    Y = torch.matmul(X, W_target.t())
    Y = torch.tanh(Y)  # No-linealidad opcional
    
    # Añadir ruido si se especifica
    if config["noise_level"] > 0:
        noise = torch.randn_like(Y) * config["noise_level"]
        Y += noise
    
    print(f"✅ Datos generados: {X.shape} -> {Y.shape}")
    return X, Y

def create_custom_model():
    """
    Crear modelo personalizado.
    
    TODO: Modifica esta función para crear tu arquitectura específica.
    """
    arch_config = PROJECT_CONFIG["architecture"]
    
    input_size = arch_config["input_size"]
    output_size = arch_config["output_size"]
    layer_type = arch_config["layer_type"]
    
    print(f"🏗️ Creando modelo {input_size}x{output_size}...")
    
    # TODO: Personaliza tu modelo aquí
    if layer_type == "MZI":
        model = MZILayer(
            in_features=input_size,
            out_features=output_size,
            device=device
        )
    elif layer_type == "MZIBlock":
        model = MZIBlockLinear(
            in_features=input_size,
            out_features=output_size,
            mode="usv",  # TODO: Cambia modo si necesario
            device=device
        )
    else:
        # TODO: Añade otros tipos de modelo si necesario
        raise ValueError(f"Tipo de modelo no soportado: {layer_type}")
    
    # Mostrar información del modelo
    if hasattr(model, 'n_mzis'):
        print(f"   🔗 MZIs físicos: {model.n_mzis}")
        print(f"   🌊 Phase shifters: {model.get_phase_shifter_count()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   🔢 Parámetros totales: {total_params}")
    
    return model

# ===================================================================
# FUNCIONES DE ENTRENAMIENTO - GENERALMENTE NO NECESITAS MODIFICAR
# ===================================================================

def train_model(model, X_train, Y_train, X_test=None, Y_test=None):
    """Entrenar el modelo."""
    train_config = PROJECT_CONFIG["training"]
    
    # Configurar optimizador
    if train_config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    elif train_config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=train_config["learning_rate"])
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=train_config["learning_rate"])
    
    # Función de pérdida - TODO: Cambia si necesitas otra pérdida
    criterion = nn.MSELoss()  # Para regresión
    # criterion = nn.CrossEntropyLoss()  # Para clasificación
    
    # Scheduler opcional
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.8)
    
    epochs = train_config["epochs"]
    
    # Historial para análisis
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [] if X_test is not None else None,
        "learning_rate": []
    }
    
    print(f"🔥 Iniciando entrenamiento:")
    print(f"   Épocas: {epochs}")
    print(f"   Learning rate: {train_config['learning_rate']}")
    print(f"   Optimizer: {train_config['optimizer']}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        
        # Forward pass
        train_pred = model(X_train)
        train_loss = criterion(train_pred, Y_train)
        
        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # Evaluación en test set si está disponible
        test_loss = None
        if X_test is not None:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, Y_test)
                scheduler.step(test_loss)
        else:
            scheduler.step(train_loss)
        
        # Guardar historial
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss.item())
        if test_loss is not None:
            history["test_loss"].append(test_loss.item())
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        
        # Mostrar progreso
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            lr = optimizer.param_groups[0]['lr']
            test_str = f", Test={test_loss.item():.6f}" if test_loss else ""
            print(f"   Época {epoch:3d}: Train={train_loss.item():.6f}{test_str}, LR={lr:.6f}")
        
        # Early stopping opcional
        current_loss = test_loss.item() if test_loss else train_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
        
    print(f"✅ Entrenamiento completado!")
    print(f"   Loss final: {history['train_loss'][-1]:.6f}")
    
    return history

def analyze_results(model, history):
    """Analizar resultados del entrenamiento."""
    print("\n🔬 ANÁLISIS DE RESULTADOS")
    print("=" * 50)
    
    # Extraer phase shifters
    extractor = PhaseShifterExtractor(verbose=False)
    phase_shifters = extractor.extract(model)
    
    # Mostrar resumen
    extractor.print_summary(phase_shifters)
    
    # Mostrar convergencia
    final_loss = history["train_loss"][-1]
    initial_loss = history["train_loss"][0]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100
    
    print(f"\n📈 CONVERGENCIA:")
    print(f"   Loss inicial: {initial_loss:.6f}")
    print(f"   Loss final: {final_loss:.6f}")
    print(f"   Mejora: {improvement:.1f}%")
    
    # Calidad del entrenamiento
    if improvement > 90:
        quality = "Excelente"
    elif improvement > 70:
        quality = "Buena"
    elif improvement > 50:
        quality = "Aceptable"
    else:
        quality = "Necesita mejoras"
    
    print(f"   Calidad: {quality}")
    
    return phase_shifters

def save_results(model, history, phase_shifters):
    """Guardar resultados del proyecto."""
    project_name = PROJECT_CONFIG["project_name"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Crear directorio de resultados
    results_dir = project_root / "results" / "my_projects" / f"{project_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Guardando resultados en: {results_dir}")
    
    # Guardar configuración del proyecto
    with open(results_dir / "project_config.json", 'w') as f:
        json.dump(PROJECT_CONFIG, f, indent=2)
    
    # Guardar phase shifters
    extractor = PhaseShifterExtractor(verbose=False)
    extractor.save_to_file(phase_shifters, str(results_dir / "phase_shifters.json"))
    
    # Guardar historial de entrenamiento
    with open(results_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Guardar modelo
    torch.save(model.state_dict(), results_dir / "model_weights.pth")
    
    # Crear resumen
    summary = {
        "project_name": project_name,
        "timestamp": timestamp,
        "final_loss": history["train_loss"][-1],
        "epochs_trained": len(history["epoch"]),
        "total_phase_shifters": phase_shifters["summary"]["total_phase_shifters"],
        "architecture": PROJECT_CONFIG["architecture"],
        "training_config": PROJECT_CONFIG["training"]
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Archivos guardados:")
    print(f"   📄 project_config.json - Configuración del proyecto")
    print(f"   📄 phase_shifters.json - Valores θ y φ entrenados")
    print(f"   📄 training_history.json - Historial de entrenamiento")
    print(f"   📄 model_weights.pth - Pesos del modelo")
    print(f"   📄 summary.json - Resumen ejecutivo")
    
    return results_dir

# ===================================================================
# FUNCIÓN PRINCIPAL - MODIFICA SI NECESARIO
# ===================================================================

def main():
    """
    Función principal del script.
    
    TODO: Modifica esta función si necesitas un flujo diferente.
    """
    print("🌟" * 40)
    print("🌟  PLANTILLA BÁSICA - ONN PERSONALIZADA  🌟")
    print("🌟" * 40)
    
    project_name = PROJECT_CONFIG["project_name"]
    print(f"📚 Proyecto: {project_name}")
    print(f"📄 Descripción: {PROJECT_CONFIG['description']}")
    
    try:
        # PASO 1: Generar datos
        print(f"\n{'='*60}")
        print("📊 PASO 1: GENERACIÓN DE DATOS")
        print('='*60)
        X, Y = generate_training_data()
        
        # Dividir datos en train/test
        test_ratio = PROJECT_CONFIG["data"]["test_ratio"]
        n_test = int(len(X) * test_ratio)
        n_train = len(X) - n_test
        
        # Mezclar datos
        perm = torch.randperm(len(X))
        X, Y = X[perm], Y[perm]
        
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_train:], Y[n_train:]
        
        print(f"✅ Datos divididos: {n_train} train, {n_test} test")
        
        # PASO 2: Crear modelo
        print(f"\n{'='*60}")
        print("🏗️ PASO 2: CREACIÓN DEL MODELO")
        print('='*60)
        model = create_custom_model()
        
        # PASO 3: Entrenar
        print(f"\n{'='*60}")
        print("🔥 PASO 3: ENTRENAMIENTO")
        print('='*60)
        history = train_model(model, X_train, Y_train, X_test, Y_test)
        
        # PASO 4: Analizar resultados
        print(f"\n{'='*60}")
        print("🔬 PASO 4: ANÁLISIS")
        print('='*60)
        phase_shifters = analyze_results(model, history)
        
        # PASO 5: Guardar resultados
        print(f"\n{'='*60}")
        print("💾 PASO 5: GUARDAR RESULTADOS")
        print('='*60)
        results_dir = save_results(model, history, phase_shifters)
        
        # RESUMEN FINAL
        print(f"\n🎉 ¡PROYECTO COMPLETADO EXITOSAMENTE!")
        print(f"✅ Modelo entrenado y optimizado")
        print(f"✅ Phase shifters extraídos")
        print(f"✅ Resultados guardados en: {results_dir}")
        
        print(f"\n🎯 PRÓXIMOS PASOS:")
        print(f"1. Revisar los archivos en {results_dir}")
        print(f"2. Usar phase_shifters.json para inferencia")
        print(f"3. Implementar en hardware si es necesario")
        
        return results_dir
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===================================================================
# PUNTO DE ENTRADA
# ===================================================================

if __name__ == "__main__":
    """
    TODO: Añade aquí cualquier configuración adicional si es necesario.
    """
    main()

# ===================================================================
# INSTRUCCIONES PARA PERSONALIZAR
# ===================================================================

"""
CÓMO PERSONALIZAR ESTA PLANTILLA:

1. 📝 CONFIGURACIÓN:
   - Modifica PROJECT_CONFIG con los parámetros de tu proyecto
   - Cambia project_name, description, tamaños de red, etc.

2. 📊 DATOS:
   - Modifica generate_training_data() para tu caso específico
   - Opciones: cargar desde archivo, generar sintéticos, etc.

3. 🏗️ MODELO:
   - Modifica create_custom_model() si necesitas arquitectura especial
   - Añade capas adicionales, cambiar tipos de MZI, etc.

4. 🔥 ENTRENAMIENTO:
   - Modifica train_model() si necesitas lógica especial
   - Cambiar función de pérdida, añadir regularización, etc.

5. ✅ EJECUCIÓN:
   - Copia este archivo a scripts/tu_proyecto/mi_script.py
   - Modifica según tus necesidades
   - Ejecuta: python scripts/tu_proyecto/mi_script.py

EJEMPLO DE USO:
   cd /workspaces/OpticalCI
   cp scripts/templates/basic_onn_template.py scripts/mi_experimento/mi_red.py
   # [editar mi_red.py según tus necesidades]
   python scripts/mi_experimento/mi_red.py
"""