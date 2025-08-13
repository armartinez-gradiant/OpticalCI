"""
Herramientas de Visualización - OpticalCI Utils
==============================================

Utilidades para visualizar resultados y análisis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def plot_phase_shifter_evolution(history, save_path=None):
    """Graficar evolución de phase shifters durante entrenamiento."""
    if 'theta_values' not in history or 'phi_values' not in history:
        print("❌ Historial no contiene valores de phase shifters")
        return
    
    theta_history = np.array(history['theta_values'])
    phi_history = np.array(history['phi_values'])
    epochs = history['epoch']
    
    n_mzis = theta_history.shape[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Theta evolution
    for i in range(n_mzis):
        ax1.plot(epochs, np.degrees(theta_history[:, i]), 
                label=f'θ_{i}', linewidth=2, alpha=0.8)
    
    ax1.set_title('Evolución de Phase Shifters θ (Theta)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Theta (grados)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Phi evolution  
    for i in range(n_mzis):
        ax2.plot(epochs, np.degrees(phi_history[:, i]), 
                label=f'φ_{i}', linewidth=2, alpha=0.8)
    
    ax2.set_title('Evolución de Phase Shifters φ (Phi)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Phi (grados)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado: {save_path}")
    
    plt.show()

def plot_training_metrics(history, save_path=None):
    """Graficar métricas de entrenamiento."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = history['epoch']
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'test_loss' in history and history['test_loss']:
        ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Evolución del Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('LR')
        ax2.grid(True, alpha=0.3)
    
    # Loss en escala log
    ax3.semilogy(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'test_loss' in history and history['test_loss']:
        ax3.semilogy(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax3.set_title('Loss (Escala Log)')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Log Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Mejora relativa
    initial_loss = history['train_loss'][0]
    relative_improvement = [(initial_loss - loss) / initial_loss * 100 
                           for loss in history['train_loss']]
    ax4.plot(epochs, relative_improvement, 'purple', linewidth=2)
    ax4.set_title('Mejora Relativa')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Mejora (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado: {save_path}")
    
    plt.show()
