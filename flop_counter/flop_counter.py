"""
Implementación Python pura del contador de FLOPs.
Esta versión funciona sin dependencias de C++ como fallback.
"""

import functools
import threading
import time
import sys
from typing import List, Optional, Any, Dict, Tuple
from contextlib import contextmanager

class OperationInfo:
    """Información sobre una operación registrada"""
    
    def __init__(self, name: str, input_shapes: List[List[int]], 
                 output_shapes: List[int], flops: int, 
                 execution_time_ms: float = 0.0, 
                 library_source: str = "unknown"):
        self.name = name
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.flops = flops
        self.execution_time_ms = execution_time_ms
        self.library_source = library_source
    
    def __repr__(self):
        return f"<OperationInfo name='{self.name}' flops={self.flops} library='{self.library_source}'>"

class FLOPCounterPython:
    """
    Contador de FLOPs implementado en Python puro.
    Proporciona funcionalidad completa sin dependencias de C++.
    """
    
    def __init__(self):
        self.total_flops = 0
        self.operations = []
        self.is_active = False
        self.lock = threading.Lock()
        self._interceptors_active = False
        self._original_functions = {}
        
        # Calculadoras de FLOPs
        self.flop_calculators = self._initialize_calculators()
    
    def _initialize_calculators(self) -> Dict[str, callable]:
        """Inicializa las calculadoras de FLOPs para diferentes operaciones"""
        calculators = {}
        
        # Operaciones de álgebra lineal
        calculators.update({
            'matmul': self._calculate_matmul_flops,
            'dot': self._calculate_matmul_flops,
            'mm': self._calculate_matmul_flops,
            'bmm': self._calculate_matmul_flops,
            'tensordot': self._calculate_matmul_flops,
        })
        
        # Operaciones elementwise
        elementwise_ops = [
            'add', 'subtract', 'multiply', 'divide', 'power',
            'sqrt', 'exp', 'log', 'sin', 'cos', 'tanh',
            'relu', 'sigmoid', 'softmax', 'sub', 'mul', 'div', 'pow'
        ]
        for op in elementwise_ops:
            calculators[op] = self._calculate_elementwise_flops
        
        # Operaciones de reducción  
        reduction_ops = [
            'sum', 'mean', 'max', 'min', 'std', 'var',
            'reduce_sum', 'reduce_mean', 'reduce_max', 'reduce_min'
        ]
        for op in reduction_ops:
            calculators[op] = self._calculate_reduction_flops
        
        # Operaciones específicas de frameworks
        calculators.update({
            'F.linear': self._calculate_matmul_flops,
            'F.relu': self._calculate_elementwise_flops,
            'F.softmax': self._calculate_elementwise_flops,
            'F.dropout': self._calculate_elementwise_flops,
            'F.conv2d': self._calculate_conv2d_flops,
            'linear': self._calculate_matmul_flops,
            'conv2d': self._calculate_conv2d_flops,
        })
        
        return calculators
    
    def start_counting(self):
        """Inicia el conteo de FLOPs"""
        with self.lock:
            self.is_active = True
            self._activate_interceptors()
    
    def stop_counting(self):
        """Detiene el conteo de FLOPs"""
        with self.lock:
            self.is_active = False
            self._deactivate_interceptors()
    
    def reset(self):
        """Resetea el contador"""
        with self.lock:
            self.total_flops = 0
            self.operations.clear()
    
    def get_total_flops(self) -> int:
        """Obtiene el total de FLOPs contados"""
        return self.total_flops
    
    def get_operations(self) -> List[Dict]:
        """
        Obtiene lista detallada de operaciones.
        
        Returns:
            List[Dict]: Lista de operaciones con sus detalles
        """
        with self.lock:
            return [
                {
                    'name': op.name,
                    'input_shapes': op.input_shapes,
                    'output_shapes': op.output_shapes,
                    'flops': op.flops,
                    'execution_time_ms': op.execution_time_ms,
                    'library_source': op.library_source
                }
                for op in self.operations
            ]
    
    def record_operation(self, op_name: str, input_shapes: List[List[int]], 
                        output_shapes: List[int], library: str = "unknown",
                        execution_time: float = 0.0):
        """Registra una operación manualmente"""
        if not self.is_active:
            return
        
        # Calcular FLOPs
        flops = self._calculate_operation_flops(op_name, input_shapes, output_shapes)
        
        # Crear info de operación
        op_info = OperationInfo(
            name=op_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            flops=flops,
            execution_time_ms=execution_time,
            library_source=library
        )
        
        with self.lock:
            self.operations.append(op_info)
            self.total_flops += flops
    
    def _calculate_operation_flops(self, op_name: str, 
                                  input_shapes: List[List[int]], 
                                  output_shapes: List[int]) -> int:
        """Calcula FLOPs para una operación específica"""
        calculator = self.flop_calculators.get(op_name)
        
        if calculator:
            return calculator(input_shapes, output_shapes)
        else:
            # Estimación genérica basada en elementos de salida
            output_elements = 1
            for dim in output_shapes:
                output_elements *= dim
            return output_elements
    
    def _calculate_matmul_flops(self, input_shapes: List[List[int]], 
                               output_shapes: List[int]) -> int:
        """Calcula FLOPs para multiplicación de matrices"""
        if len(input_shapes) < 2:
            return 0
        
        shape_a, shape_b = input_shapes[0], input_shapes[1]
        
        if len(shape_a) < 2 or len(shape_b) < 2:
            return 0
        
        # Para A (batch x m x k) * B (batch x k x n) = C (batch x m x n)
        m = shape_a[-2]
        k = shape_a[-1]
        n = shape_b[-1]
        
        # Calcular batch size
        batch_size = 1
        for i in range(len(shape_a) - 2):
            batch_size *= shape_a[i]
        
        return 2 * batch_size * m * k * n  # 2 FLOPs por multiplicación-suma
    
    def _calculate_conv2d_flops(self, input_shapes: List[List[int]], 
                               output_shapes: List[int]) -> int:
        """Calcula FLOPs para convolución 2D"""
        if len(input_shapes) < 2 or len(output_shapes) < 4:
            # Estimación genérica si no tenemos suficiente información
            output_elements = 1
            for dim in output_shapes:
                output_elements *= dim
            return output_elements * 10  # Estimación conservadora
        
        input_shape = input_shapes[0]  # [batch, in_channels, h_in, w_in]
        kernel_shape = input_shapes[1] if len(input_shapes) > 1 else [64, input_shape[1], 3, 3]
        
        if len(input_shape) < 4 or len(kernel_shape) < 4:
            return self._calculate_elementwise_flops(input_shapes, output_shapes) * 10
        
        # output: [batch, out_channels, h_out, w_out]
        batch_size = output_shapes[0]
        out_channels = output_shapes[1]
        h_out = output_shapes[2]
        w_out = output_shapes[3]
        
        in_channels = kernel_shape[1]
        k_h = kernel_shape[2]
        k_w = kernel_shape[3]
        
        return batch_size * out_channels * h_out * w_out * in_channels * k_h * k_w * 2
    
    def _calculate_elementwise_flops(self, input_shapes: List[List[int]], 
                                   output_shapes: List[int]) -> int:
        """Calcula FLOPs para operaciones elementwise"""
        # Usar el tamaño de salida para operaciones elementwise
        output_elements = 1
        for dim in output_shapes:
            output_elements *= dim
        return output_elements
    
    def _calculate_reduction_flops(self, input_shapes: List[List[int]], 
                                 output_shapes: List[int]) -> int:
        """Calcula FLOPs para operaciones de reducción"""
        if not input_shapes:
            return 0
        
        input_elements = 1
        for dim in input_shapes[0]:
            input_elements *= dim
        return input_elements
    
    def print_summary(self):
        """Imprime un resumen detallado"""
        with self.lock:
            print("\n=== FLOP Counter Summary (Python Backend) ===")
            print(f"Total FLOPs: {self.total_flops:,}")
            print(f"Total Operations: {len(self.operations)}")
            
            if not self.operations:
                print("No operations recorded.")
                return
            
            # Agrupar por tipo de operación
            flops_by_op = {}
            count_by_op = {}
            time_by_op = {}
            
            for op in self.operations:
                name = op.name
                flops_by_op[name] = flops_by_op.get(name, 0) + op.flops
                count_by_op[name] = count_by_op.get(name, 0) + 1
                time_by_op[name] = time_by_op.get(name, 0.0) + op.execution_time_ms
            
            print("\nBreakdown by operation:")
            print(f"{'Operation':<15} {'Count':<8} {'Total FLOPs':<15} {'Avg FLOPs':<12} {'Total Time':<12}")
            print("-" * 72)
            
            for op_name in sorted(flops_by_op.keys()):
                total_op_flops = flops_by_op[op_name]
                op_count = count_by_op[op_name]
                total_time = time_by_op[op_name]
                avg_flops = total_op_flops / op_count
                
                print(f"{op_name:<15} {op_count:<8} {total_op_flops:<15,} {avg_flops:<12,.0f} {total_time:<12.2f}ms")
            
            print()
    
    def _activate_interceptors(self):
        """Activa los interceptores para librerías populares"""
        if self._interceptors_active:
            return
        
        self._interceptors_active = True
        
        # NumPy
        try:
            self._intercept_numpy()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to intercept NumPy: {e}", ImportWarning)
        
        # PyTorch
        try:
            self._intercept_torch()
        except Exception:
            pass  # PyTorch no disponible o error al interceptar
        
        # TensorFlow - solo interceptar si ya está importado
        try:
            if 'tensorflow' in sys.modules:
                self._intercept_tensorflow()
        except Exception:
            pass  # TensorFlow no disponible o error al interceptar
    
    def _deactivate_interceptors(self):
        """Desactiva los interceptores"""
        if not self._interceptors_active:
            return
        
        # Restaurar funciones originales
        for module_name, functions in self._original_functions.items():
            try:
                if module_name == 'numpy':
                    import numpy as np
                    module = np
                elif module_name == 'torch':
                    import torch
                    module = torch
                elif module_name == 'torch.nn.functional':
                    import torch.nn.functional as F
                    module = F
                elif module_name == 'tensorflow':
                    import tensorflow as tf
                    module = tf
                else:
                    continue
                
                for func_name, original_func in functions.items():
                    setattr(module, func_name, original_func)
            except ImportError:
                continue
        
        self._original_functions.clear()
        self._interceptors_active = False
    
    def _intercept_numpy(self):
        """Intercepta funciones de NumPy"""
        try:
            import numpy as np
            
            if 'numpy' not in self._original_functions:
                self._original_functions['numpy'] = {}
            
            # Funciones de álgebra lineal (prioridad alta - siempre interceptar)
            critical_functions = ['matmul', 'dot', 'inner', 'outer', 'tensordot', 'vdot', 'einsum']
            
            for func_name in critical_functions:
                if hasattr(np, func_name):
                    try:
                        original_func = getattr(np, func_name)
                        self._original_functions['numpy'][func_name] = original_func
                        wrapped_func = self._create_numpy_wrapper(func_name, original_func)
                        setattr(np, func_name, wrapped_func)
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Failed to intercept np.{func_name}: {e}", ImportWarning)
            
            # Operaciones de reducción (estas son funciones, no ufuncs generalmente)
            reduction_functions = [
                'sum', 'mean', 'std', 'var', 'max', 'min',
                'prod', 'median', 'percentile', 'average',
                'nansum', 'nanmean', 'nanstd', 'nanvar',
                'nanmax', 'nanmin', 'nanmedian'
            ]
            
            for func_name in reduction_functions:
                if hasattr(np, func_name):
                    try:
                        original_func = getattr(np, func_name)
                        
                        # Verificar si es un ufunc (tiene métodos reduce y accumulate)
                        is_ufunc = (hasattr(original_func, 'reduce') and 
                                   hasattr(original_func, 'accumulate') and
                                   hasattr(original_func, 'at'))
                        
                        if is_ufunc:
                            # Es un ufunc, no lo interceptamos
                            continue
                        
                        self._original_functions['numpy'][func_name] = original_func
                        wrapped_func = self._create_numpy_wrapper(func_name, original_func)
                        setattr(np, func_name, wrapped_func)
                    except Exception:
                        # Si falla una función específica, continuar con las demás
                        continue
            
            # Interceptar operaciones de álgebra lineal del módulo linalg
            if hasattr(np, 'linalg'):
                linalg_functions = ['norm', 'det', 'inv', 'solve', 'eig', 'svd', 'qr', 'cholesky']
                for func_name in linalg_functions:
                    if hasattr(np.linalg, func_name):
                        try:
                            original_func = getattr(np.linalg, func_name)
                            self._original_functions['numpy'][f'linalg.{func_name}'] = original_func
                            
                            wrapped_func = self._create_numpy_wrapper(f'linalg.{func_name}', original_func)
                            setattr(np.linalg, func_name, wrapped_func)
                        except Exception:
                            continue
        
        except ImportError:
            pass
    
    def _create_numpy_wrapper(self, func_name: str, original_func: callable):
        """Crea un wrapper para funciones de NumPy"""
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            # Solo contar si el contador está activo
            if not self.is_active:
                return original_func(*args, **kwargs)
            
            start_time = time.perf_counter()
            result = original_func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # ms
            
            # Extraer shapes
            input_shapes = []
            for arg in args:
                if hasattr(arg, 'shape'):
                    input_shapes.append(list(arg.shape))
            
            output_shape = list(result.shape) if hasattr(result, 'shape') else [1]
            
            # Registrar operación
            self.record_operation(
                op_name=func_name,
                input_shapes=input_shapes,
                output_shapes=output_shape,
                library="numpy",
                execution_time=execution_time
            )
            
            return result
        
        return wrapper
    
    def _intercept_torch(self):
        """Intercepta funciones de PyTorch"""
        # Solo interceptar si PyTorch está disponible
        if 'torch' not in sys.modules:
            try:
                import torch
            except ImportError:
                return
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Interceptar funciones de torch
            if 'torch' not in self._original_functions:
                self._original_functions['torch'] = {}
            
            torch_functions = [
                'matmul', 'mm', 'bmm', 'add', 'sub', 'mul', 'div',
                'pow', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tanh',
                'sum', 'mean', 'max', 'min'
            ]
            
            for func_name in torch_functions:
                if hasattr(torch, func_name):
                    try:
                        original_func = getattr(torch, func_name)
                        self._original_functions['torch'][func_name] = original_func
                        
                        wrapped_func = self._create_torch_wrapper(func_name, original_func)
                        setattr(torch, func_name, wrapped_func)
                    except Exception:
                        continue
            
            # Interceptar torch.nn.functional
            if 'torch.nn.functional' not in self._original_functions:
                self._original_functions['torch.nn.functional'] = {}
            
            f_functions = ['linear', 'conv2d', 'relu', 'softmax', 'dropout']
            for func_name in f_functions:
                if hasattr(F, func_name):
                    try:
                        original_func = getattr(F, func_name)
                        self._original_functions['torch.nn.functional'][func_name] = original_func
                        
                        wrapped_func = self._create_torch_wrapper(f"F.{func_name}", original_func)
                        setattr(F, func_name, wrapped_func)
                    except Exception:
                        continue
        
        except Exception:
            pass
    
    def _create_torch_wrapper(self, func_name: str, original_func: callable):
        """Crea un wrapper para funciones de PyTorch"""
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = original_func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # ms
            
            # Extraer shapes
            input_shapes = []
            for arg in args:
                if hasattr(arg, 'shape'):
                    input_shapes.append(list(arg.shape))
            
            output_shape = list(result.shape) if hasattr(result, 'shape') else [1]
            
            # Registrar operación
            self.record_operation(
                op_name=func_name,
                input_shapes=input_shapes,
                output_shapes=output_shape,
                library="torch",
                execution_time=execution_time
            )
            
            return result
        
        return wrapper
    
    def _intercept_tensorflow(self):
        """Intercepta funciones de TensorFlow"""
        # Solo interceptar si TensorFlow ya está importado
        if 'tensorflow' not in sys.modules:
            return
        
        try:
            import tensorflow as tf
            
            if 'tensorflow' not in self._original_functions:
                self._original_functions['tensorflow'] = {}
            
            tf_functions = [
                'matmul', 'add', 'subtract', 'multiply', 'divide',
                'pow', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tanh',
                'reduce_sum', 'reduce_mean', 'reduce_max', 'reduce_min'
            ]
            
            for func_name in tf_functions:
                if hasattr(tf, func_name):
                    try:
                        original_func = getattr(tf, func_name)
                        self._original_functions['tensorflow'][func_name] = original_func
                        
                        wrapped_func = self._create_tensorflow_wrapper(func_name, original_func)
                        setattr(tf, func_name, wrapped_func)
                    except Exception:
                        # Si falla interceptar una función específica, continuar con las demás
                        continue
        
        except Exception:
            # Si TensorFlow causa problemas, simplemente no lo interceptamos
            pass
    
    def _create_tensorflow_wrapper(self, func_name: str, original_func: callable):
        """Crea un wrapper para funciones de TensorFlow"""
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = original_func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # ms
            
            # Extraer shapes
            input_shapes = []
            for arg in args:
                if hasattr(arg, 'shape'):
                    input_shapes.append(list(arg.shape))
            
            output_shape = list(result.shape) if hasattr(result, 'shape') else [1]
            
            # Registrar operación
            self.record_operation(
                op_name=func_name,
                input_shapes=input_shapes,
                output_shapes=output_shape,
                library="tensorflow",
                execution_time=execution_time
            )
            
            return result
        
        return wrapper
    
    @contextmanager
    def count_flops(self):
        """Context manager para contar FLOPs"""
        self.start_counting()
        try:
            yield self
        finally:
            self.stop_counting()
    
    def __del__(self):
        """Destructor"""
        if hasattr(self, '_interceptors_active') and self._interceptors_active:
            self._deactivate_interceptors()

# Instancia global del contador
_global_counter = None

def get_global_counter() -> FLOPCounterPython:
    """Obtiene o crea el contador global"""
    global _global_counter
    if _global_counter is None:
        _global_counter = FLOPCounterPython()
    return _global_counter

# Funciones de conveniencia
def start_counting():
    """Inicia el conteo global de FLOPs"""
    get_global_counter().start_counting()

def stop_counting():
    """Detiene el conteo global de FLOPs"""
    get_global_counter().stop_counting()

def reset_counting():
    """Resetea el contador global"""
    get_global_counter().reset()

def get_total_flops() -> int:
    """Obtiene el total de FLOPs del contador global"""
    return get_global_counter().get_total_flops()

def print_summary():
    """Imprime resumen del contador global"""
    get_global_counter().print_summary()

@contextmanager
def count_flops():
    """Context manager para contar FLOPs globalmente"""
    with get_global_counter().count_flops() as counter:
        yield counter