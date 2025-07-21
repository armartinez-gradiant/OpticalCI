#!/usr/bin/env python3
"""
Ejemplo de entrenamiento completo con PtONN-TESTS
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchonn.layers import MZILayer, MZIBlockLinear
from torchonn.models import ONNBaseModel

class ONNClassifier(ONNBaseModel):
    """Clasificador ONN completo."""
    
    def __init__(self, input_size, hidden_sizes=[16, 8], num_classes=3, mode="usv"):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Capas ocultas
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layer = MZIBlockLinear(
                in_features=prev_size,
                out_features=hidden_size,
                mode=mode,
                device=self.device
            )
            self.layers.append(layer)
            prev_size = hidden_size
        
        # Capa de salida
        self.output_layer = MZILayer(
            in_features=prev_size,
            out_features=num_classes,
            device=self.device
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            if self.training:
                x = self.dropout(x)
        
        x = self.output_layer(x)
        return x

def create_dataset(n_samples=1000, n_features=20, n_classes=3):
    """Crear dataset sintético."""
    print(f"📊 Creando dataset: {n_samples} muestras, {n_features} características, {n_classes} clases")
    
    # ✅ FIX: Ajustar parámetros según n_features
    n_informative = min(n_features - 2, max(2, int(n_features * 0.7)))  # 70% informativas
    n_redundant = min(2, n_features - n_informative - 1)  # Al menos 1 libre
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Normalizar datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convertir a tensores
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Test: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_test, y_test, epochs=50):
    """Entrenar el modelo."""
    print(f"\n🎯 Entrenando modelo por {epochs} épocas...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluación
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test).sum().item() / y_test.size(0)
                test_accuracies.append(accuracy)
                
                print(f"   Época {epoch:3d}: Loss = {loss.item():.4f}, Test Acc = {accuracy:.4f}")
    
    return train_losses, test_accuracies

def compare_modes():
    """Comparar diferentes modos de MZI Block."""
    print("\n🔬 Comparando modos de MZI Block...")
    
    # Crear dataset común
    X_train, X_test, y_train, y_test = create_dataset(n_samples=500, n_features=10)
    
    modes = ["usv", "weight", "phase"]
    results = {}
    
    for mode in modes:
        print(f"\n🧪 Probando modo: {mode}")
        
        # Crear modelo
        model = ONNClassifier(
            input_size=10,
            hidden_sizes=[8, 6],
            num_classes=3,
            mode=mode
        )
        
        print(f"   Parámetros: {sum(p.numel() for p in model.parameters())}")
        
        # Entrenar
        train_losses, test_accs = train_model(
            model, X_train, y_train, X_test, y_test, epochs=30
        )
        
        # Evaluación final
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            final_accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
        results[mode] = {
            'final_accuracy': final_accuracy,
            'train_losses': train_losses,
            'test_accuracies': test_accs
        }
        
        print(f"   Precisión final: {final_accuracy:.4f}")
    
    return results

def test_persistence():
    """Test de guardado y carga de modelos."""
    print("\n💾 Probando persistencia de modelos...")
    
    # Crear y entrenar modelo
    X_train, X_test, y_train, y_test = create_dataset(n_samples=200, n_features=8)
    model = ONNClassifier(input_size=8, hidden_sizes=[6], num_classes=3)
    
    # Entrenamiento rápido
    train_model(model, X_train, y_train, X_test, y_test, epochs=10)
    
    # Guardar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': 8,
        'hidden_sizes': [6],
        'num_classes': 3
    }, 'onn_model.pth')
    
    # Cargar modelo
    checkpoint = torch.load('onn_model.pth')
    new_model = ONNClassifier(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['hidden_sizes'],
        num_classes=checkpoint['num_classes']
    )
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Verificar que son iguales
    model.eval()
    new_model.eval()
    
    with torch.no_grad():
        test_input = torch.randn(5, 8)
        output1 = model(test_input)
        output2 = new_model(test_input)
        
        if torch.allclose(output1, output2, atol=1e-6):
            print("   ✅ Modelo guardado y cargado correctamente")
        else:
            print("   ❌ Error en guardado/carga")
    
    print("   📁 Modelo guardado en: onn_model.pth")

def main():
    """Función principal."""
    print("🌟 Ejemplo de Entrenamiento Completo - PtONN-TESTS")
    print("=" * 60)
    
    # Configurar semilla para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Comparar modos
    results = compare_modes()
    
    # 2. Test de persistencia
    test_persistence()
    
    # 3. Resumen final
    print("\n📈 Resumen de Resultados:")
    print("-" * 40)
    for mode, result in results.items():
        print(f"   {mode:6s}: {result['final_accuracy']:.4f} precisión")
    
    print("\n🎉 ¡Entrenamiento completo finalizado!")
    print("\nPrueba ejecutar:")
    print("   python example_training.py")

if __name__ == "__main__":
    main()