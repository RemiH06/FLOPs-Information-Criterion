# El contenido de este archivo ya fue creado anteriormente en cpp_interface.py
# Este archivo es idéntico al anterior cpp_interface.py

"""
Interfaz mejorada para el backend C++ del contador de FLOPs.
Proporciona una API Python más limpia sobre el módulo C++.
"""

import functools
import sys
from typing import List, Optional, Any, Dict
from contextlib import contextmanager
import threading

try:
    import flop_counter_cpp
    _cpp_available = True
except ImportError:
    _cpp_available = False
    flop_counter_cpp = None

class FLOPCounterCPP:
    """
    Wrapper Python para el contador de FLOPs implementado en C++.
    Proporciona la misma interfaz que FLOPCounterPython pero con mejor rendimiento.
    """
    
    def __init__(self):
        if not _cpp_available:
            raise ImportError("Backend C++ no disponible. Usar FLOPCounterPython instead.")
        
        self._counter = flop_counter_cpp.FLOPCounter()
        self._interceptors_active = False
        self._original_functions = {}
        self._lock = threading.Lock()
    
    def start_counting(self):
        """Inicia el conteo de FLOPs"""
        with self._lock:
            self._counter.start_counting()
            self._activate_interceptors()
    
    def stop_counting(self):
        """Detiene el conteo de FLOPs"""
        with self._lock:
            self._counter.stop_counting()
            self._deactivate_interceptors()
    
    def reset(self):
        """Resetea el contador"""
        with self._lock:
            self._counter.reset()
    
    def get_total_flops(self) -> int:
        """Obtiene el total de FLOPs contados"""
        return self._counter.get_total_flops()
    
    def get_operations(self) -> List[Dict]:
        """
        Obtiene lista detallada de todas las operaciones registradas.
        
        Returns:
            List[Dict]: Lista de operaciones con detalles
        """
        cpp_ops = self._counter.get_operations()
        return [
            {
                'name': op.name,
                'input_shapes': op.input_shapes,
                'output_shapes': op.output_shapes,
                'flops': op.flops,
                'execution_time_ms': op.execution_time_ms,
                'library_source': op.library_source
            }
            for op in cpp_ops
        ]
    
    def print_summary(self):
        """Imprime un resumen detallado"""
        self._counter.print_summary()
    
    def record_operation(self, op_name: str, input_shapes: List[List[int]], 
                        output_shapes: List[int], library: str = "unknown",
                        execution_time: float = 0.0):
        """Registra una operación manualmente"""
        # Convertir a formato esperado por C++
        flattened_inputs = []
        for shape in input_shapes:
            flattened_inputs.extend(shape)
        
        self._counter.record_operation(
            op_name, flattened_inputs, output_shapes, library, execution_time
        )
    
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
            pass
        
        # TensorFlow - solo si ya está importado
        try:
            if 'tensorflow' in sys.modules:
                self._intercept_tensorflow()
        except Exception:
            pass
    
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
                elif module_name == 'tensorflow':
                    import tensorflow as tf
                    module = tf
                else:
                    continue
                
                for func_name, original_func in functions.items():
                    if '.' in func_name:
                        # Manejar funciones anidadas como F.conv2d
                        parts = func_name.split('.')
                        if parts[0] == 'F' and module_name == 'torch':
                            import torch.nn.functional as F
                            setattr(F, parts[1], original_func)
                    else:
                        setattr(module, func_name, original_func)
            except ImportError:
                continue
        
        self._original_functions.clear()
        self._interceptors_active = False
    
    def _intercept_numpy(self):
        """Intercepta funciones de NumPy con mejor rendimiento"""
        try:
            import numpy as np
            
            if 'numpy' not in self._original_functions:
                self._original_functions['numpy'] = {}
            
            # Funciones a interceptar
            np_functions = [
                'dot', 'matmul', 'add', 'subtract', 'multiply', 'divide',
                'power', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tanh',
                'sum', 'mean', 'max', 'min'
            ]
            
            for func_name in np_functions:
                if hasattr(np, func_name):
                    original_func = getattr(np, func_name)
                    self._original_functions['numpy'][func_name] = original_func
                    
                    def create_wrapper(fname, orig_func):
                        @functools.wraps(orig_func)
                        def wrapper(*args, **kwargs):
                            # Medir tiempo de ejecución
                            import time
                            start_time = time.perf_counter()
                            
                            result = orig_func(*args, **kwargs)
                            
                            end_time = time.perf_counter()
                            execution_time = (end_time - start_time) * 1000  # ms
                            
                            # Extraer shapes de manera más eficiente
                            input_shapes = []
                            for arg in args:
                                if hasattr(arg, 'shape'):
                                    input_shapes.append(list(arg.shape))
                            
                            output_shape = list(result.shape) if hasattr(result, 'shape') else [1]
                            
                            # Usar el backend C++ directamente
                            self._counter.record_operation(
                                fname, 
                                [item for sublist in input_shapes for item in sublist], 
                                output_shape, 
                                "numpy", 
                                execution_time
                            )
                            
                            return result
                        return wrapper
                    
                    wrapped_func = create_wrapper(func_name, original_func)
                    setattr(np, func_name, wrapped_func)
        
        except ImportError:
            pass
    
    def _intercept_torch(self):
        """Intercepta funciones de PyTorch"""
        try:
            import torch
            import torch.nn.functional as F
            
            if 'torch' not in self._original_functions:
                self._original_functions['torch'] = {}
            
            # Funciones de tensor
            torch_functions = [
                'matmul', 'mm', 'bmm', 'add', 'sub', 'mul', 'div',
                'pow', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tanh',
                'sum', 'mean', 'max', 'min'
            ]
            
            for func_name in torch_functions:
                if hasattr(torch, func_name):
                    original_func = getattr(torch, func_name)
                    self._original_functions['torch'][func_name] = original_func
                    
                    def create_wrapper(fname, orig_func):
                        @functools.wraps(orig_func)
                        def wrapper(*args, **kwargs):
                            import time
                            start_time = time.perf_counter()
                            
                            result = orig_func(*args, **kwargs)
                            
                            end_time = time.perf_counter()
                            execution_time = (end_time - start_time) * 1000
                            
                            input_shapes = []
                            for arg in args:
                                if hasattr(arg, 'shape'):
                                    input_shapes.append(list(arg.shape))
                            
                            output_shape = list(result.shape) if hasattr(result, 'shape') else [1]
                            
                            self._counter.record_operation(
                                fname,
                                [item for sublist in input_shapes for item in sublist],
                                output_shape,
                                "torch",
                                execution_time
                            )
                            
                            return result
                        return wrapper
                    
                    wrapped_func = create_wrapper(func_name, original_func)
                    setattr(torch, func_name, wrapped_func)
            
            # Interceptar torch.nn.functional
            f_functions = ['conv2d', 'linear', 'relu', 'softmax', 'dropout']
            for func_name in f_functions:
                if hasattr(F, func_name):
                    original_func = getattr(F, func_name)
                    self._original_functions['torch'][f'F.{func_name}'] = original_func
                    
                    def create_f_wrapper(fname, orig_func):
                        @functools.wraps(orig_func)
                        def wrapper(*args, **kwargs):
                            import time
                            start_time = time.perf_counter()
                            
                            result = orig_func(*args, **kwargs)
                            
                            end_time = time.perf_counter()
                            execution_time = (end_time - start_time) * 1000
                            
                            input_shapes = []
                            for arg in args:
                                if hasattr(arg, 'shape'):
                                    input_shapes.append(list(arg.shape))
                            
                            output_shape = list(result.shape) if hasattr(result, 'shape') else [1]
                            
                            self._counter.record_operation(
                                f"F.{fname}",
                                [item for sublist in input_shapes for item in sublist],
                                output_shape,
                                "torch",
                                execution_time
                            )
                            
                            return result
                        return wrapper
                    
                    wrapped_func = create_f_wrapper(func_name, original_func)
                    setattr(F, func_name, wrapped_func)
        
        except ImportError:
            pass
    
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
                        
                        def create_wrapper(fname, orig_func):
                            @functools.wraps(orig_func)
                            def wrapper(*args, **kwargs):
                                import time
                                start_time = time.perf_counter()
                                
                                result = orig_func(*args, **kwargs)
                                
                                end_time = time.perf_counter()
                                execution_time = (end_time - start_time) * 1000
                                
                                input_shapes = []
                                for arg in args:
                                    if hasattr(arg, 'shape'):
                                        input_shapes.append(list(arg.shape))
                                
                                output_shape = list(result.shape) if hasattr(result, 'shape') else [1]
                                
                                self._counter.record_operation(
                                    fname,
                                    [item for sublist in input_shapes for item in sublist],
                                    output_shape,
                                    "tensorflow",
                                    execution_time
                                )
                                
                                return result
                            return wrapper
                        
                        wrapped_func = create_wrapper(func_name, original_func)
                        setattr(tf, func_name, wrapped_func)
                    except Exception:
                        continue
        
        except Exception:
            pass
    
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

# Función de conveniencia para crear instancias
def create_cpp_counter() -> Optional[FLOPCounterCPP]:
    """
    Crea una nueva instancia del contador C++ si está disponible.
    
    Returns:
        FLOPCounterCPP o None si el backend C++ no está disponible
    """
    if _cpp_available:
        return FLOPCounterCPP()
    return None