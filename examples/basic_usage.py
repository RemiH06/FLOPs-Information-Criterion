#!/usr/bin/env python3
"""
Ejemplo básico de uso del Universal FLOP Counter
Este ejemplo demuestra cómo usar la librería con diferentes frameworks de ML.
"""

import sys
import os

# Fix para OpenMP warning en Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Añadir el directorio padre para importar nuestra librería
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import flop_counter

def test_numpy_operations():
    """Test básico con operaciones de NumPy"""
    print("=== Testing NumPy Operations ===")
    print("Note: Only high-level NumPy functions are intercepted (matmul, dot, sum, etc.)")
    print("Element-wise ufuncs (add, multiply, etc.) are not intercepted to avoid conflicts.\n")
    
    # Verificar que numpy está disponible y las funciones existen
    import numpy as np
    print(f"NumPy matmul type: {type(np.matmul)}")
    print(f"NumPy dot type: {type(np.dot)}\n")
    
    with flop_counter.count_flops() as counter:
        # Crear matrices
        A = np.random.randn(100, 50)
        B = np.random.randn(50, 75)
        
        print(f"Before operations: {counter.get_total_flops():,} FLOPs")
        
        # Multiplicación de matrices (INTERCEPTADA)
        C = np.matmul(A, B)
        print(f"  After matmul(100x50, 50x75): {counter.get_total_flops():,} FLOPs (expected ~750,000)")
        
        # Más operaciones matriciales (INTERCEPTADAS)
        D = np.dot(C.T, C)  # Producto punto
        print(f"  After dot(75x100, 100x75): {counter.get_total_flops():,} FLOPs (expected ~1,875,000)")
        
        # Operaciones de reducción (INTERCEPTADAS)
        result_sum = np.sum(D)
        print(f"  After sum: {counter.get_total_flops():,} FLOPs")
        
        result_mean = np.mean(D)
        print(f"  After mean: {counter.get_total_flops():,} FLOPs")
        
        result_std = np.std(D)
        print(f"  After std: {counter.get_total_flops():,} FLOPs")
        
    print(f"\nTotal FLOPs: {counter.get_total_flops():,}")
    counter.print_summary()

def test_pytorch_operations():
    """Test con operaciones de PyTorch"""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        print("\n=== Testing PyTorch Operations ===")
        
        with flop_counter.count_flops() as counter:
            # Crear tensores
            x = torch.randn(32, 784)  # batch_size=32, input_size=784
            
            # Red neuronal simple
            linear1 = nn.Linear(784, 256)
            linear2 = nn.Linear(256, 128)
            linear3 = nn.Linear(128, 10)
            
            # Forward pass
            h1 = F.relu(linear1(x))
            h2 = F.relu(linear2(h1))
            output = linear3(h2)
            
            # Softmax
            probs = F.softmax(output, dim=1)
            
            # Loss (simulado)
            loss = torch.sum(probs)
            
        print(f"Total FLOPs: {counter.get_total_flops():,}")
        counter.print_summary()
        
    except ImportError:
        print("PyTorch not available, skipping PyTorch tests")
    except Exception as e:
        print(f"Error in PyTorch test: {e}")

def test_tensorflow_operations():
    """Test con operaciones de TensorFlow - DESHABILITADO por conflictos"""
    print("\n=== TensorFlow Test Skipped ===")
    print("TensorFlow interception disabled to avoid library conflicts.")
    print("You can still use TensorFlow, but automatic interception may not work.")
    return
    
    # Código TensorFlow comentado para evitar conflictos
    # try:
    #     import tensorflow as tf
    #     
    #     with flop_counter.count_flops() as counter:
    #         x = tf.random.normal([32, 784])
    #         ...
    # except Exception as e:
    #     print(f"TensorFlow test failed: {e}") TensorFlow"""
    try:
        import tensorflow as tf
        
        print("\n=== Testing TensorFlow Operations ===")
        
        with flop_counter.count_flops() as counter:
            # Crear tensores
            x = tf.random.normal([32, 784])
            
            # Definir capas
            dense1 = tf.keras.layers.Dense(256, activation='relu')
            dense2 = tf.keras.layers.Dense(128, activation='relu') 
            dense3 = tf.keras.layers.Dense(10)
            
            # Forward pass
            h1 = dense1(x)
            h2 = dense2(h1)
            output = dense3(h2)
            
            # Softmax
            probs = tf.nn.softmax(output)
            
            # Operaciones adicionales
            loss = tf.reduce_sum(probs)
            
        print(f"Total FLOPs: {counter.get_total_flops():,}")
        counter.print_summary()
        
    except ImportError:
        print("TensorFlow not available, skipping TensorFlow tests")

def test_mixed_operations():
    """Test mezclando diferentes librerías"""
    print("\n=== Testing Mixed Operations ===")
    
    counter = flop_counter.FLOPCounterPython()
    
    with counter.count_flops():
        # NumPy
        a = np.random.randn(64, 64)
        b = np.random.randn(64, 64)
        c = np.matmul(a, b)
        
        try:
            import torch
            # PyTorch
            x = torch.from_numpy(c).float()
            y = torch.relu(x)
            z = torch.sum(y)
        except ImportError:
            pass
        
        # Más NumPy
        d = np.exp(c)
        result = np.mean(d)
        
    print(f"Total FLOPs: {counter.get_total_flops():,}")
    counter.print_summary()

def test_manual_recording():
    """Test de registro manual de operaciones"""
    print("\n=== Testing Manual Operation Recording ===")
    
    counter = flop_counter.FLOPCounterPython()
    counter.start_counting()
    
    # Simular una convolution 2D
    counter.record_operation(
        op_name="conv2d",
        input_shapes=[[1, 3, 224, 224], [64, 3, 7, 7]],  # input, kernel
        output_shapes=[1, 64, 112, 112],
        library="custom",
        execution_time=1.5
    )
    
    # Simular batch normalization
    counter.record_operation(
        op_name="batch_norm",
        input_shapes=[[1, 64, 112, 112]],
        output_shapes=[1, 64, 112, 112],
        library="custom",
        execution_time=0.3
    )
    
    # Simular una operación personalizada
    counter.record_operation(
        op_name="custom_attention",
        input_shapes=[[32, 512, 768], [32, 512, 768]],  # query, key
        output_shapes=[32, 512, 768],
        library="transformer",
        execution_time=2.1
    )
    
    counter.stop_counting()
    
    print(f"Total FLOPs: {counter.get_total_flops():,}")
    counter.print_summary()

def test_context_manager():
    """Test usando el context manager global"""
    print("\n=== Testing Global Context Manager ===")
    
    with flop_counter.count_flops():
        # Simular entrenamiento de una época
        for batch in range(10):
            # Simular forward pass
            x = np.random.randn(32, 784)
            W1 = np.random.randn(784, 256)
            b1 = np.random.randn(256)
            
            h1 = np.matmul(x, W1) + b1
            h1_relu = np.maximum(0, h1)  # ReLU
            
            W2 = np.random.randn(256, 10)
            b2 = np.random.randn(10)
            
            output = np.matmul(h1_relu, W2) + b2
            
            # Softmax (simulado)
            exp_output = np.exp(output)
            softmax = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            
    print(f"Total FLOPs: {flop_counter.get_total_flops():,}")
    flop_counter.print_summary()

def demonstrate_model_comparison():
    """Demostración de comparación de modelos"""
    print("\n=== Model Comparison Demo ===")
    
    def small_model(x):
        """Modelo pequeño"""
        W1 = np.random.randn(100, 50)
        W2 = np.random.randn(50, 10)
        h = np.matmul(x, W1)
        h = np.tanh(h)  # Esta operación no se cuenta (ufunc)
        output = np.matmul(h, W2)
        return output
    
    def large_model(x):
        """Modelo grande"""
        W1 = np.random.randn(100, 200)
        W2 = np.random.randn(200, 100)
        W3 = np.random.randn(100, 10)
        h1 = np.matmul(x, W1)
        h1 = np.tanh(h1)  # Esta operación no se cuenta (ufunc)
        h2 = np.matmul(h1, W2)
        h2 = np.tanh(h2)  # Esta operación no se cuenta (ufunc)
        output = np.matmul(h2, W3)
        return output
    
    x = np.random.randn(32, 100)  # batch de datos
    
    # Test modelo pequeño
    flop_counter.reset_counting()
    with flop_counter.count_flops() as counter_small:
        result_small = small_model(x)
    flops_small = counter_small.get_total_flops()
    
    # Test modelo grande
    flop_counter.reset_counting()
    with flop_counter.count_flops() as counter_large:
        result_large = large_model(x)
    flops_large = counter_large.get_total_flops()
    
    print(f"Small Model FLOPs: {flops_small:,}")
    print(f"Large Model FLOPs: {flops_large:,}")
    
    if flops_small > 0 and flops_large > 0:
        print(f"Ratio: {flops_large / flops_small:.2f}x more FLOPs")
    else:
        print("Note: Some operations (like tanh) are not intercepted as they are NumPy ufuncs")
        print("Only matmul operations were counted in this demo")

def demonstrate_efficiency_analysis():
    """Análisis de eficiencia"""
    print("\n=== Efficiency Analysis Demo ===")
    
    import time
    
    sizes = [50, 100, 200, 500]
    
    print(f"{'Size':<10} {'FLOPs':<15} {'Time (ms)':<12} {'GFLOPS/s':<12}")
    print("-" * 55)
    
    for size in sizes:
        # Crear matrices
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)
        
        # Medir FLOPs y tiempo
        flop_counter.reset_counting()
        start_time = time.perf_counter()
        
        with flop_counter.count_flops():
            C = np.matmul(A, B)
        
        end_time = time.perf_counter()
        
        flops = flop_counter.get_total_flops()
        time_ms = (end_time - start_time) * 1000
        gflops_per_sec = (flops / 1e9) / (time_ms / 1000)
        
        print(f"{size:<10} {flops:<15,} {time_ms:<12.3f} {gflops_per_sec:<12.2f}")

if __name__ == "__main__":
    print("Universal FLOP Counter - Basic Usage Example")
    print("=" * 60)
    
    # Mostrar información del backend
    print("\nBackend Information:")
    backend_info = flop_counter.get_backend_info()
    for key, value in backend_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Reset global counter
    flop_counter.reset_counting()
    
    # Ejecutar todos los tests
    test_numpy_operations()
    test_pytorch_operations()
    test_tensorflow_operations()
    test_mixed_operations()
    test_manual_recording()
    test_context_manager()
    demonstrate_model_comparison()
    demonstrate_efficiency_analysis()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    
    # Mostrar resumen final global
    print(f"\nGlobal counter total: {flop_counter.get_total_flops():,}")