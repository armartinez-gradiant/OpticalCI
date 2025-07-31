"""
Optical MNIST Benchmark - COMPLETAMENTE CORREGIDO

ERROR DE INDENTACIÓN LÍNEA 88 CORREGIDO ✅
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import warnings

from ...models import ONNBaseModel

try:
    from ..architectures.coherent_onn import CoherentONN
    COHERENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CoherentONN no disponible: {e}")
    COHERENT_AVAILABLE = False


class OpticalMNIST:
    """Benchmark MNIST para Optical Neural Networks."""
    
    def __init__(
        self,
        image_size: int = 8,
        n_classes: int = 10,
        n_samples_per_class: int = 100,
        device: Optional[Union[str, torch.device]] = None
    ):
        """Inicializar benchmark."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        self.image_size = image_size
        self.n_classes = n_classes
        self.n_samples_per_class = n_samples_per_class
        self.input_features = image_size * image_size
        
        self.train_data, self.train_labels = self._create_synthetic_mnist()
        self.test_data, self.test_labels = self._create_synthetic_mnist(train=False)
        
        print(f"🔬 OpticalMNIST Benchmark initialized:")
        print(f"   Train samples: {len(self.train_data)}")
        print(f"   Test samples: {len(self.test_data)}")
    
    def _create_synthetic_mnist(self, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crear dataset MNIST sintético.
        
        FUNCIÓN CORRECTAMENTE INDENTADA - ERROR LÍNEA 88 CORREGIDO.
        """
        n_samples = self.n_samples_per_class * self.n_classes
        if not train:
            n_samples = n_samples // 4
        
        data_list = []
        labels_list = []
        
        for class_idx in range(self.n_classes):
            n_class_samples = n_samples // self.n_classes
            
            base_pattern = torch.zeros(self.input_features)
            
            if class_idx == 0:  # "0" - círculo
                center = self.image_size // 2
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                        if abs(dist - center * 0.7) < 1.0:
                            base_pattern[i * self.image_size + j] = 1.0
            
            elif class_idx == 1:  # "1" - línea vertical
                center = self.image_size // 2
                for i in range(self.image_size):
                    base_pattern[i * self.image_size + center] = 1.0
            
            else:  # Otros dígitos
                torch.manual_seed(class_idx * 42)
                indices = torch.randperm(self.input_features)[:self.input_features // 3]
                base_pattern[indices] = 1.0
            
            for _ in range(n_class_samples):
                noise_level = 0.1 if train else 0.05
                pattern = base_pattern + torch.randn_like(base_pattern) * noise_level
                pattern = torch.clamp(pattern, 0, 1)
                
                data_list.append(pattern)
                labels_list.append(class_idx)
        
        data = torch.stack(data_list)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        perm = torch.randperm(len(data))
        data = data[perm].to(self.device)
        labels = labels[perm].to(self.device)
        
        return data, labels
    
    def create_coherent_onn(self, hidden_size: int = 16) -> CoherentONN:
        """Crear CoherentONN."""
        if not COHERENT_AVAILABLE:
            raise ImportError("CoherentONN no disponible")
        
        return CoherentONN(
            layer_sizes=[self.input_features, hidden_size, self.n_classes],
            activation_type="square_law",
            optical_power=1.0,
            use_unitary_constraints=True,
            device=self.device
        )
    
    def create_baseline_nn(self, hidden_size: int = 16) -> nn.Module:
        """Crear baseline NN."""
        return nn.Sequential(
            nn.Linear(self.input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_classes)
        ).to(self.device)
    
    def train_model(
        self, 
        model: nn.Module, 
        n_epochs: int = 10,
        learning_rate: float = 0.01,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """Entrenar modelo."""
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            n_batches = len(self.train_data) // batch_size
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.train_data))
                
                batch_data = self.train_data[start_idx:end_idx]
                batch_labels = self.train_labels[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            avg_loss = epoch_loss / n_batches
            accuracy = 100.0 * correct / total
            
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            if epoch % max(1, n_epochs // 5) == 0 or epoch == n_epochs - 1:
                print(f"   Época {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.1f}%")
        
        return history
    
    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluar modelo."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            outputs = model(self.test_data)
            _, predicted = torch.max(outputs, 1)
            total = self.test_labels.size(0)
            correct = (predicted == self.test_labels).sum().item()
        
        accuracy = 100.0 * correct / total
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def run_comparison_benchmark(
        self, 
        n_epochs: int = 10,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """Ejecutar benchmark comparativo."""
        print("🚀 Ejecutando Optical MNIST Benchmark...")
        results = {}
        
        if COHERENT_AVAILABLE:
            print("\n🔬 Testing CoherentONN...")
            try:
                start_time = time.time()
                coherent_onn = self.create_coherent_onn()
                
                coherent_history = self.train_model(
                    coherent_onn, 
                    n_epochs=n_epochs, 
                    learning_rate=learning_rate
                )
                
                coherent_results = self.evaluate_model(coherent_onn)
                training_time = time.time() - start_time
                
                results["CoherentONN"] = {
                    "training_history": coherent_history,
                    "test_results": coherent_results,
                    "training_time": training_time,
                    "final_accuracy": coherent_results["accuracy"]
                }
                
                print(f"   ✅ CoherentONN: {coherent_results['accuracy']:.1f}% accuracy")
                
            except Exception as e:
                print(f"   ❌ CoherentONN failed: {e}")
                results["CoherentONN"] = {"error": str(e)}
        
        print("\n🔧 Testing Baseline NN...")
        try:
            start_time = time.time()
            baseline_nn = self.create_baseline_nn()
            
            baseline_history = self.train_model(
                baseline_nn, 
                n_epochs=n_epochs, 
                learning_rate=learning_rate
            )
            
            baseline_results = self.evaluate_model(baseline_nn)
            training_time = time.time() - start_time
            
            results["BaselineNN"] = {
                "training_history": baseline_history,
                "test_results": baseline_results,
                "training_time": training_time,
                "final_accuracy": baseline_results["accuracy"]
            }
            
            print(f"   ✅ Baseline NN: {baseline_results['accuracy']:.1f}% accuracy")
            
        except Exception as e:
            print(f"   ❌ Baseline NN failed: {e}")
            results["BaselineNN"] = {"error": str(e)}
        
        print("\n📊 RESUMEN COMPARATIVO:")
        for model_name, result in results.items():
            if "error" not in result:
                acc = result["final_accuracy"]
                time_taken = result["training_time"]
                print(f"   {model_name}: {acc:.1f}% accuracy ({time_taken:.1f}s)")
            else:
                print(f"   {model_name}: ERROR - {result['error']}")
        
        return results


def run_quick_demo(image_size: int = 4, n_epochs: int = 5) -> Dict[str, Any]:
    """Demo rápido."""
    if not COHERENT_AVAILABLE:
        print("❌ CoherentONN no disponible")
        return {}
    
    print("🚀 Quick Optical MNIST Demo")
    benchmark = OpticalMNIST(image_size=image_size)
    results = benchmark.run_comparison_benchmark(n_epochs=n_epochs, learning_rate=0.01)
    return results


OpticalMNISTBenchmark = OpticalMNIST

def create_optical_mnist_benchmark(**kwargs):
    """Factory function."""
    return OpticalMNIST(**kwargs)


if __name__ == "__main__":
    results = run_quick_demo(image_size=4, n_epochs=3)
    print("\n🎉 Demo completed!")