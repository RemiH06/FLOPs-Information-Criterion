#!/usr/bin/env python3
"""
Ejemplo de uso de la función de alto nivel count_model_flops()

Versión en .py para leer de corrido en consola
"""

import sys
import os

# Fix para OpenMP warning en Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Añadir el directorio padre para importar nuestra librería
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import flop_counter

print("="*70)
print("UNIVERSAL FLOP COUNTER - HIGH LEVEL API EXAMPLES")
print("="*70)

# ============================================================================
# EJEMPLO 1: Función Simple de NumPy
# ============================================================================

def simple_numpy_model(x):
    """Modelo simple con NumPy"""
    W1 = np.random.randn(100, 50)
    W2 = np.random.randn(50, 10)
    
    h = np.matmul(x, W1)
    output = np.matmul(h, W2)
    return output

print("\n" + "="*70)
print("EJEMPLO 1: Función NumPy Simple")
print("="*70)

x = np.random.randn(32, 100)

result = flop_counter.count_model_flops(
    model=simple_numpy_model,
    input_data=x,
    framework='numpy',
    verbose=True
)

print(f"\nResultado retornado:")
print(f"  Total FLOPs: {result['total_flops']:,}")
print(f"  Tiempo: {result['execution_time_ms']:.2f} ms")

# ============================================================================
# EJEMPLO 2: Modelo PyTorch
# ============================================================================

import torch
import torch.nn as nn

print("\n" + "="*70)
print("EJEMPLO 2: Modelo PyTorch")
print("="*70)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet()
x = torch.randn(32, 784)

result = flop_counter.count_model_flops(
    model=model,
    input_data=x,
    verbose=True
)

print(f"\nDetalles del resultado:")
print(f"  Framework detectado: {result['framework']}")
print(f"  Parámetros del modelo: {result['model_info'].get('total_parameters', 'N/A'):,}")

# ============================================================================
# EJEMPLO 3: Comparación de Modelos
# ============================================================================

print("\n" + "="*70)
print("EJEMPLO 3: Comparación de Modelos")
print("="*70)

def model_small(x):
    """Modelo pequeño"""
    W = np.random.randn(100, 10)
    return np.matmul(x, W)

def model_medium(x):
    """Modelo mediano"""
    W1 = np.random.randn(100, 50)
    W2 = np.random.randn(50, 10)
    h = np.matmul(x, W1)
    return np.matmul(h, W2)

def model_large(x):
    """Modelo grande"""
    W1 = np.random.randn(100, 200)
    W2 = np.random.randn(200, 100)
    W3 = np.random.randn(100, 10)
    h1 = np.matmul(x, W1)
    h2 = np.matmul(h1, W2)
    return np.matmul(h2, W3)

x = np.random.randn(32, 100)

models = {
    'Small': (model_small, x),
    'Medium': (model_medium, x),
    'Large': (model_large, x)
}

results = flop_counter.compare_models(models, framework='numpy', verbose=True)

# ============================================================================
# EJEMPLO 4: Uso sin verbose (solo obtener FLOPs)
# ============================================================================

print("\n" + "="*70)
print("EJEMPLO 4: Uso Simple (solo FLOPs)")
print("="*70)

def my_model(x):
    W = np.random.randn(50, 25)
    return np.matmul(x, W)

x = np.random.randn(16, 50)

# Obtener solo el número de FLOPs
flops = flop_counter.count_model_flops(
    model=my_model,
    input_data=x,
    verbose=False,
    return_details=False
)

print(f"FLOPs calculados: {flops:,}")

# ============================================================================
# EJEMPLO 5: Modelo con PyTorch más complejo
# ============================================================================

import torch
import torch.nn as nn

print("\n" + "="*70)
print("EJEMPLO 5: Red Convolucional PyTorch")
print("="*70)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()
x = torch.randn(1, 3, 32, 32)  # Imagen 32x32 RGB

result = flop_counter.count_model_flops(
    model=model,
    input_data=x,
    verbose=True
)

# ============================================================================
# EJEMPLO 6: Análisis de Eficiencia
# ============================================================================

print("\n" + "="*70)
print("EJEMPLO 6: Análisis de Eficiencia por FLOPs")
print("="*70)

def analyze_efficiency(model_fn, input_data, model_name):
    """Analiza la eficiencia de un modelo."""
    result = flop_counter.count_model_flops(
        model=model_fn,
        input_data=input_data,
        verbose=False,
        return_details=True
    )
    
    flops = result['total_flops']
    time_ms = result['execution_time_ms']
    
    # Calcular métricas
    throughput = (flops / (time_ms / 1000)) / 1e9  # GFLOPS/s
    
    # Obtener tamaño de salida
    output_shape = result.get('output_shape', None)
    
    print(f"\n{model_name}:")
    print(f"  FLOPs:      {flops:,}")
    print(f"  Time:       {time_ms:.2f} ms")
    print(f"  Throughput: {throughput:.2f} GFLOPS/s")
    if output_shape:
        print(f"  Output:     {output_shape}")
    
    return result

# Crear varios modelos de diferentes tamaños
input_data = np.random.randn(32, 128)

def tiny_model(x):
    return np.matmul(x, np.random.randn(128, 10))

def small_model(x):
    h = np.matmul(x, np.random.randn(128, 64))
    return np.matmul(h, np.random.randn(64, 10))

def medium_model(x):
    h1 = np.matmul(x, np.random.randn(128, 128))
    h2 = np.matmul(h1, np.random.randn(128, 64))
    return np.matmul(h2, np.random.randn(64, 10))

results = {
    'Tiny': analyze_efficiency(tiny_model, input_data, 'Tiny Model'),
    'Small': analyze_efficiency(small_model, input_data, 'Small Model'),
    'Medium': analyze_efficiency(medium_model, input_data, 'Medium Model')
}

# Comparar eficiencia
print(f"\n{'Model':<15} {'FLOPs':<15} {'Time (ms)':<12} {'Efficiency':<15}")
print("-" * 60)
for name, result in results.items():
    flops = result['total_flops']
    time_ms = result['execution_time_ms']
    efficiency = flops / time_ms if time_ms > 0 else 0
    print(f"{name:<15} {flops:<15,} {time_ms:<12.2f} {efficiency:<15,.0f}")

# ============================================================================
# EJEMPLO 7: Detección Automática de Framework
# ============================================================================

print("\n" + "="*70)
print("EJEMPLO 7: Detección Automática de Framework")
print("="*70)

# NumPy
def numpy_fn(x):
    return np.matmul(x, np.random.randn(50, 10))

x_np = np.random.randn(32, 50)

result_np = flop_counter.count_model_flops(
    model=numpy_fn,
    input_data=x_np,
    framework='auto',  # Detectar automáticamente
    verbose=False
)

print(f"NumPy model - Framework detectado: {result_np['framework']}")
print(f"FLOPs: {result_np['total_flops']:,}")

import torch

def torch_fn(x):
    W = torch.randn(50, 10)
    return torch.matmul(x, W)

x_torch = torch.randn(32, 50)

result_torch = flop_counter.count_model_flops(
    model=torch_fn,
    input_data=x_torch,
    framework='auto',  # Detectar automáticamente
    verbose=False
)

print(f"\nPyTorch model - Framework detectado: {result_torch['framework']}")
print(f"FLOPs: {result_torch['total_flops']:,}")

# ============================================================================
# EJEMPLO 8: Breakdown Detallado de Operaciones
# ============================================================================

print("\n" + "="*70)
print("EJEMPLO 8: Breakdown Detallado de Operaciones")
print("="*70)

def complex_model(x):
    """Modelo con múltiples tipos de operaciones."""
    # Capas fully connected
    h1 = np.matmul(x, np.random.randn(100, 256))
    h2 = np.matmul(h1, np.random.randn(256, 128))
    h3 = np.matmul(h2, np.random.randn(128, 10))
    
    # Operaciones de reducción
    mean_h3 = np.mean(h3)
    std_h3 = np.std(h3)
    sum_h3 = np.sum(h3)
    
    return h3

x = np.random.randn(32, 100)

result = flop_counter.count_model_flops(
    model=complex_model,
    input_data=x,
    verbose=False,
    return_details=True
)

print("\nBreakdown por tipo de operación:")
print(f"{'Operación':<20} {'FLOPs':<15} {'% del Total':<12}")
print("-" * 50)

total = result['total_flops']
for op_name, op_flops in sorted(result['flops_by_operation'].items(), 
                                key=lambda x: x[1], reverse=True):
    percentage = (op_flops / total * 100) if total > 0 else 0
    print(f"{op_name:<20} {op_flops:<15,} {percentage:>10.1f}%")

print(f"\nTotal: {total:,} FLOPs")