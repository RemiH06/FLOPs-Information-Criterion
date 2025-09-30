"""
Universal FLOP Counter
=====================
Una librería universal para contar FLOPs en cualquier modelo de machine learning,
independientemente del framework utilizado (NumPy, PyTorch, TensorFlow, etc.).

Características principales:
- Intercepción automática de operaciones de ML
- Soporte para múltiples frameworks simultáneamente
- Conteo preciso de FLOPs basado en formas de tensores
- Interfaz simple y pythónica
- Backend optimizado en C++

Uso básico:
    >>> import flop_counter
    >>> with flop_counter.count_flops() as counter:
    ...     # Código de ML aquí
    ...     result = model(data)
    >>> print(f"FLOPs utilizados: {counter.get_total_flops()}")

Uso de alto nivel:
    >>> import flop_counter
    >>> result = flop_counter.count_model_flops(model, input_data)
    >>> print(f"FLOPs: {result['total_flops']:,}")

Autores: Remi Heredia
Versión: 0.2.0
Licencia: MIT
"""

from .flop_counter import (
    FLOPCounterPython,
    get_global_counter,
    start_counting,
    stop_counting,
    reset_counting,
    get_total_flops,
    print_summary,
    count_flops
)

# Importar función de alto nivel
from .model_profiler import (
    count_model_flops,
    compare_models
)

# Intentar importar el backend C++ optimizado
try:
    import flop_counter_cpp
    _cpp_backend_available = True
    
    # Usar el backend C++ si está disponible
    from .cpp_interface import FLOPCounterCPP
    FLOPCounter = FLOPCounterCPP
    
except ImportError:
    _cpp_backend_available = False
    # Fallback al backend Python puro
    FLOPCounter = FLOPCounterPython

__version__ = "0.2.0"
__author__ = "Remi Heredia"
__email__ = "tu.email@ejemplo.com"

# Exportar símbolos principales
__all__ = [
    # Clases principales
    'FLOPCounter',
    'FLOPCounterPython',
    
    # Funciones de conveniencia (bajo nivel)
    'count_flops',
    'start_counting',
    'stop_counting',
    'reset_counting',
    'get_total_flops',
    'print_summary',
    'get_global_counter',
    
    # Funciones de alto nivel
    'count_model_flops',
    'compare_models',
    
    # Metadatos
    '__version__',
    '__author__',
    '__email__',
]

def get_backend_info():
    """
    Retorna información sobre el backend actualmente en uso.
    
    Returns:
        dict: Información del backend incluyendo tipo y capacidades
    """
    return {
        'backend_type': 'cpp' if _cpp_backend_available else 'python',
        'cpp_available': _cpp_backend_available,
        'version': __version__,
        'supported_libraries': ['numpy', 'torch', 'tensorflow', 'custom'],
        'threading_support': True,
        'precision': 'int64'
    }

def check_installation():
    """
    Verifica que la instalación esté funcionando correctamente.
    
    Returns:
        bool: True si todo funciona correctamente
    """
    try:
        # Test básico de funcionamiento
        import numpy as np
        
        with count_flops() as counter:
            a = np.array([[1, 2], [3, 4]])
            b = np.array([[5, 6], [7, 8]])
            c = np.matmul(a, b)
        
        flops = counter.get_total_flops()
        
        if flops > 0:
            print(f"✓ Instalación verificada exitosamente")
            print(f"✓ Backend: {get_backend_info()['backend_type']}")
            print(f"✓ FLOPs de test: {flops}")
            return True
        else:
            print("✗ La instalación parece tener problemas - no se contaron FLOPs")
            return False
            
    except Exception as e:
        print(f"✗ Error en la verificación: {e}")
        return False

# Configuración inicial
def _initialize():
    """Inicialización interna del módulo"""
    import warnings
    
    if not _cpp_backend_available:
        warnings.warn(
            "Backend C++ no disponible. Usando implementación Python pura. "
            "Para mejor rendimiento, recompila con pybind11.",
            ImportWarning
        )

# Ejecutar inicialización
_initialize()