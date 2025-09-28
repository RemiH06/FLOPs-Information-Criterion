#!/usr/bin/env python3
"""
Tests unitarios para Universal FLOP Counter
"""

import unittest
import numpy as np
import sys
import os

# Añadir el directorio padre para importar la librería
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flop_counter

class TestFLOPCounterBasic(unittest.TestCase):
    """Tests básicos de funcionalidad"""
    
    def setUp(self):
        """Configuración antes de cada test"""
        flop_counter.reset_counting()
    
    def tearDown(self):
        """Limpieza después de cada test"""
        flop_counter.stop_counting()
        flop_counter.reset_counting()
    
    def test_installation(self):
        """Test de que la instalación funciona"""
        self.assertTrue(flop_counter.check_installation())
    
    def test_basic_counting(self):
        """Test básico de conteo"""
        with flop_counter.count_flops() as counter:
            a = np.array([[1, 2], [3, 4]])
            b = np.array([[5, 6], [7, 8]])
            c = np.matmul(a, b)
        
        total_flops = counter.get_total_flops()
        self.assertGreater(total_flops, 0, "Deberían contarse algunos FLOPs")
    
    def test_context_manager(self):
        """Test del context manager global"""
        initial_flops = flop_counter.get_total_flops()
        
        with flop_counter.count_flops():
            a = np.random.randn(10, 10)
            b = np.random.randn(10, 10)
            c = np.matmul(a, b)
        
        final_flops = flop_counter.get_total_flops()
        self.assertGreater(final_flops, initial_flops, "FLOPs deberían incrementar")
    
    def test_reset_functionality(self):
        """Test de la funcionalidad de reset"""
        # Realizar algunas operaciones
        with flop_counter.count_flops():
            a = np.random.randn(5, 5)
            b = np.matmul(a, a)
        
        flops_before_reset = flop_counter.get_total_flops()
        self.assertGreater(flops_before_reset, 0)
        
        # Reset
        flop_counter.reset_counting()
        flops_after_reset = flop_counter.get_total_flops()
        self.assertEqual(flops_after_reset, 0, "FLOPs deberían ser 0 después del reset")
    
    def test_start_stop_manually(self):
        """Test de iniciar/parar manualmente"""
        flop_counter.start_counting()
        
        a = np.random.randn(8, 8)
        b = np.random.randn(8, 8)
        c = np.matmul(a, b)
        
        flop_counter.stop_counting()
        
        flops = flop_counter.get_total_flops()
        self.assertGreater(flops, 0, "Deberían contarse FLOPs con start/stop manual")

class TestNumpyOperations(unittest.TestCase):
    """Tests específicos para operaciones de NumPy"""
    
    def setUp(self):
        flop_counter.reset_counting()
    
    def tearDown(self):
        flop_counter.stop_counting()
        flop_counter.reset_counting()
    
    def test_matmul_operations(self):
        """Test de operaciones de multiplicación de matrices"""
        with flop_counter.count_flops() as counter:
            # Matrices pequeñas para verificación fácil
            a = np.random.randn(3, 4)
            b = np.random.randn(4, 5)
            c = np.matmul(a, b)  # 3*4*5*2 = 120 FLOPs teóricos
        
        flops = counter.get_total_flops()
        
        # Verificar que se contaron FLOPs (puede no ser exactamente 120 debido a overhead)
        self.assertGreater(flops, 100, f"Se esperaban ~120 FLOPs, se obtuvieron {flops}")
        self.assertLess(flops, 200, f"FLOPs parecen demasiado altos: {flops}")
    
    def test_elementwise_operations(self):
        """Test de operaciones elemento por elemento"""
        with flop_counter.count_flops() as counter:
            a = np.random.randn(100)
            b = np.random.randn(100)
            
            c = np.add(a, b)      # 100 FLOPs
            d = np.multiply(c, 2)  # 100 FLOPs
            e = np.tanh(d)        # 100 FLOPs
        
        flops = counter.get_total_flops()
        self.assertGreater(flops, 200, f"Se esperaban ~300 FLOPs, se obtuvieron {flops}")
    
    def test_reduction_operations(self):
        """Test de operaciones de reducción"""
        with flop_counter.count_flops() as counter:
            a = np.random.randn(50, 50)
            
            total = np.sum(a)     # 2500 elementos -> ~2500 FLOPs
            mean_val = np.mean(a) # Similar a sum + división
        
        flops = counter.get_total_flops()
        self.assertGreater(flops, 2000, f"Se esperaban varios miles de FLOPs, se obtuvieron {flops}")

@unittest.skipUnless('torch' in sys.modules or __name__ == '__main__', 
                    "PyTorch no disponible")
class TestPyTorchOperations(unittest.TestCase):
    """Tests para operaciones de PyTorch"""
    
    def setUp(self):
        try:
            import torch
            self.torch = torch
            flop_counter.reset_counting()
        except ImportError:
            self.skipTest("PyTorch no disponible")
    
    def tearDown(self):
        flop_counter.stop_counting()
        flop_counter.reset_counting()
    
    def test_torch_matmul(self):
        """Test multiplicación de matrices en PyTorch"""
        with flop_counter.count_flops() as counter:
            a = self.torch.randn(10, 20)
            b = self.torch.randn(20, 15)
            c = self.torch.matmul(a, b)
        
        flops = counter.get_total_flops()
        expected_flops = 10 * 20 * 15 * 2  # 6000
        
        self.assertGreater(flops, expected_flops * 0.8, 
                          f"FLOPs muy bajos: {flops}, esperados ~{expected_flops}")
        self.assertLess(flops, expected_flops * 1.5,
                       f"FLOPs muy altos: {flops}, esperados ~{expected_flops}")
    
    def test_torch_nn_operations(self):
        """Test operaciones de torch.nn"""
        import torch.nn.functional as F
        
        with flop_counter.count_flops() as counter:
            x = self.torch.randn(32, 100)  # batch_size=32, features=100
            
            # Linear transformation (simulated)
            weight = self.torch.randn(50, 100)
            y = self.torch.matmul(x, weight.t())  # 32*100*50*2 FLOPs
            
            # Activation
            z = F.relu(y)  # 32*50 FLOPs
        
        flops = counter.get_total_flops()
        self.assertGreater(flops, 100000, f"Se esperaban >100k FLOPs, se obtuvieron {flops}")

@unittest.skipUnless('tensorflow' in sys.modules or __name__ == '__main__',
                    "TensorFlow no disponible")
class TestTensorFlowOperations(unittest.TestCase):
    """Tests para operaciones de TensorFlow"""
    
    def setUp(self):
        try:
            import tensorflow as tf
            self.tf = tf
            flop_counter.reset_counting()
        except ImportError:
            self.skipTest("TensorFlow no disponible")
    
    def tearDown(self):
        flop_counter.stop_counting() 
        flop_counter.reset_counting()
    
    def test_tf_matmul(self):
        """Test multiplicación de matrices en TensorFlow"""
        with flop_counter.count_flops() as counter:
            a = self.tf.random.normal([8, 12])
            b = self.tf.random.normal([12, 6])
            c = self.tf.matmul(a, b)
        
        flops = counter.get_total_flops()
        expected_flops = 8 * 12 * 6 * 2  # 1152
        
        self.assertGreater(flops, expected_flops * 0.5,
                          f"FLOPs muy bajos: {flops}, esperados ~{expected_flops}")

class TestManualRecording(unittest.TestCase):
    """Tests para registro manual de operaciones"""
    
    def setUp(self):
        self.counter = flop_counter.FLOPCounterPython()
    
    def test_manual_operation_recording(self):
        """Test de registro manual de operaciones"""
        self.counter.start()
        
        # Registrar operación personalizada
        self.counter.record_operation(
            op_name="custom_conv2d",
            input_shapes=[[1, 3, 224, 224], [64, 3, 7, 7]],
            output_shapes=[1, 64, 112, 112],
            library="custom_framework",
            execution_time=1.5
        )
        
        self.counter.stop()
        
        flops = self.counter.get_total_flops()
        self.assertGreater(flops, 0, "Operación manual debería generar FLOPs")
        
        ops = self.counter.get_operations() if hasattr(self.counter, 'get_operations') else []
        if ops:
            self.assertEqual(len(ops), 1, "Debería haber exactamente 1 operación registrada")
            self.assertEqual(ops[0]['name'], "custom_conv2d")
            self.assertEqual(ops[0]['library_source'], "custom_framework")

class TestMultipleFrameworks(unittest.TestCase):
    """Tests mezclando múltiples frameworks"""
    
    def setUp(self):
        flop_counter.reset_counting()
    
    def tearDown(self):
        flop_counter.stop_counting()
        flop_counter.reset_counting()
    
    def test_mixed_numpy_torch(self):
        """Test mezclando NumPy y PyTorch"""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch no disponible")
        
        with flop_counter.count_flops() as counter:
            # NumPy
            a_np = np.random.randn(10, 10)
            b_np = np.random.randn(10, 10)
            c_np = np.matmul(a_np, b_np)
            
            # PyTorch
            a_torch = torch.randn(10, 10)
            b_torch = torch.randn(10, 10)
            c_torch = torch.matmul(a_torch, b_torch)
            
            # Conversión entre frameworks
            d_np = c_torch.numpy()
            result = np.sum(d_np)
        
        flops = counter.get_total_flops()
        self.assertGreater(flops, 1500, f"Operaciones mixtas deberían generar >1500 FLOPs, obtuvimos {flops}")

class TestPerformance(unittest.TestCase):
    """Tests de rendimiento y overhead"""
    
    def setUp(self):
        flop_counter.reset_counting()
    
    def tearDown(self):
        flop_counter.stop_counting()
        flop_counter.reset_counting()
    
    def test_overhead_measurement(self):
        """Medir overhead del contador"""
        import time
        
        # Operación sin contador
        start_time = time.perf_counter()
        for _ in range(100):
            a = np.random.randn(50, 50)
            b = np.random.randn(50, 50) 
            c = np.matmul(a, b)
        no_counter_time = time.perf_counter() - start_time
        
        # La misma operación con contador
        start_time = time.perf_counter()
        with flop_counter.count_flops():
            for _ in range(100):
                a = np.random.randn(50, 50)
                b = np.random.randn(50, 50)
                c = np.matmul(a, b)
        with_counter_time = time.perf_counter() - start_time
        
        overhead_ratio = with_counter_time / no_counter_time
        print(f"\nOverhead ratio: {overhead_ratio:.2f}x")
        print(f"Sin contador: {no_counter_time:.4f}s")
        print(f"Con contador: {with_counter_time:.4f}s")
        
        # El overhead no debería ser más de 3x en el peor caso
        self.assertLess(overhead_ratio, 3.0, f"Overhead demasiado alto: {overhead_ratio:.2f}x")
    
    def test_large_scale_operations(self):
        """Test con operaciones de gran escala"""
        with flop_counter.count_flops() as counter:
            # Operaciones grandes
            a = np.random.randn(500, 500)
            b = np.random.randn(500, 500)
            c = np.matmul(a, b)  # 500*500*500*2 = 250M FLOPs
            
            # Algunas operaciones adicionales
            d = np.tanh(c)       # 500*500 = 250K FLOPs
            e = np.sum(d)        # 500*500 = 250K FLOPs
        
        flops = counter.get_total_flops()
        expected_min = 250_000_000  # 250M FLOPs mínimo
        
        self.assertGreater(flops, expected_min, 
                          f"Se esperaban >{expected_min:,} FLOPs, se obtuvieron {flops:,}")
        print(f"\nOperaciones de gran escala: {flops:,} FLOPs contados")

class TestEdgeCases(unittest.TestCase):
    """Tests para casos extremos y edge cases"""
    
    def setUp(self):
        self.counter = flop_counter.FLOPCounterPython()
    
    def test_empty_operations(self):
        """Test con operaciones vacías"""
        self.counter.start()
        
        # Operaciones con arrays vacíos
        a = np.array([])
        b = np.array([])
        
        # Esto puede generar warnings, pero no debería crashear
        try:
            c = np.add(a, b)
        except:
            pass  # Es OK si falla, solo no debe crashear el contador
        
        self.counter.stop()
        
        # El contador debería manejar esto gracefully
        flops = self.counter.get_total_flops()
        self.assertGreaterEqual(flops, 0, "FLOPs no deberían ser negativos")
    
    def test_scalar_operations(self):
        """Test con operaciones escalares"""
        self.counter.start()
        
        # Operaciones escalares
        a = np.array(5.0)
        b = np.array(3.0)
        c = np.add(a, b)
        d = np.multiply(c, 2.0)
        
        self.counter.stop()
        
        flops = self.counter.get_total_flops()
        self.assertGreater(flops, 0, "Operaciones escalares deberían contar FLOPs")
    
    def test_multidimensional_operations(self):
        """Test con operaciones multidimensionales complejas"""
        self.counter.start()
        
        # Arrays 4D (como en deep learning)
        a = np.random.randn(2, 3, 4, 5)  # batch, channels, height, width
        b = np.random.randn(2, 3, 4, 5)
        
        c = np.add(a, b)
        d = np.mean(c, axis=(2, 3))  # Reducir dimensiones espaciales
        
        self.counter.stop()
        
        flops = self.counter.get_total_flops()
        self.assertGreater(flops, 100, "Operaciones 4D deberían generar FLOPs significativos")
    
    def test_multiple_starts_stops(self):
        """Test con múltiples start/stop"""
        # Primer periodo
        self.counter.start()
        a = np.random.randn(10, 10)
        b = np.matmul(a, a)
        self.counter.stop()
        
        flops_1 = self.counter.get_total_flops()
        
        # Segundo periodo (sin reset)
        self.counter.start()
        c = np.random.randn(15, 15)
        d = np.matmul(c, c)
        self.counter.stop()
        
        flops_2 = self.counter.get_total_flops()
        
        # Los FLOPs deberían acumularse
        self.assertGreater(flops_2, flops_1, "Los FLOPs deberían acumularse entre periodos")
    
    def test_concurrent_operations(self):
        """Test básico de thread safety"""
        import threading
        import time
        
        results = []
        
        def worker():
            with flop_counter.count_flops():
                for _ in range(10):
                    a = np.random.randn(20, 20)
                    b = np.matmul(a, a)
                    time.sleep(0.001)  # Simular trabajo
                
                results.append(flop_counter.get_total_flops())
        
        # Ejecutar múltiples threads
        threads = []
        for _ in range(3):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Todos los threads deberían haber contado algunos FLOPs
        for result in results:
            self.assertGreater(result, 1000, f"Thread debería haber contado >1000 FLOPs, obtuvo {result}")

class TestBackendCompatibility(unittest.TestCase):
    """Tests de compatibilidad entre backends"""
    
    def test_backend_info(self):
        """Test de información del backend"""
        info = flop_counter.get_backend_info()
        
        self.assertIn('backend_type', info)
        self.assertIn('cpp_available', info)
        self.assertIn('version', info)
        self.assertIn('supported_libraries', info)
        
        self.assertIn(info['backend_type'], ['cpp', 'python'])
        self.assertIsInstance(info['cpp_available'], bool)
        self.assertIsInstance(info['supported_libraries'], list)
        
        print(f"\nBackend info: {info}")
    
    def test_python_backend_fallback(self):
        """Test que el backend Python funcione como fallback"""
        # Crear instancia específica de Python backend
        python_counter = flop_counter.FLOPCounterPython()
        
        python_counter.start()
        a = np.random.randn(5, 5)
        b = np.matmul(a, a)
        python_counter.stop()
        
        flops = python_counter.get_total_flops()
        self.assertGreater(flops, 0, "Backend Python debería funcionar independientemente")

def run_performance_benchmark():
    """Benchmark completo de rendimiento"""
    print("\n" + "="*60)
    print("BENCHMARK DE RENDIMIENTO")
    print("="*60)
    
    import time
    
    # Configuración del benchmark
    sizes = [(100, 100), (200, 200), (500, 500)]
    iterations = [100, 50, 10]
    
    for size, iters in zip(sizes, iterations):
        print(f"\nTest {size[0]}x{size[1]} matrices, {iters} iteraciones:")
        
        # Sin contador
        start_time = time.perf_counter()
        for _ in range(iters):
            a = np.random.randn(*size)
            b = np.random.randn(*size)
            c = np.matmul(a, b)
        baseline_time = time.perf_counter() - start_time
        
        # Con contador
        start_time = time.perf_counter()
        with flop_counter.count_flops() as counter:
            for _ in range(iters):
                a = np.random.randn(*size)
                b = np.random.randn(*size)
                c = np.matmul(a, b)
        counter_time = time.perf_counter() - start_time
        
        total_flops = counter.get_total_flops()
        overhead = (counter_time / baseline_time - 1) * 100
        
        print(f"  Tiempo base:     {baseline_time:.4f}s")
        print(f"  Tiempo contador: {counter_time:.4f}s")
        print(f"  Overhead:        {overhead:.1f}%")
        print(f"  FLOPs totales:   {total_flops:,}")
        print(f"  FLOPs/segundo:   {total_flops/counter_time:,.0f}")

if __name__ == '__main__':
    # Configurar el runner de tests
    import argparse
    
    parser = argparse.ArgumentParser(description='Tests para Universal FLOP Counter')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Ejecutar benchmark de rendimiento')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Output verbose')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_performance_benchmark()
    
    # Configurar nivel de verbosidad
    verbosity = 2 if args.verbose else 1
    
    # Ejecutar tests
    unittest.main(argv=[''], verbosity=verbosity, exit=False)
    
    # Ejecutar benchmark si se solicitó
    if args.benchmark:
        run_performance_benchmark()