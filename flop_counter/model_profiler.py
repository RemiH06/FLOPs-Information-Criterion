"""
Model Profiler - Función de alto nivel para contar FLOPs de modelos completos
"""

import sys
from typing import Union, Optional, Dict, Any, Tuple, Callable
import warnings

try:
    from .flop_counter import FLOPCounterPython, get_global_counter
except ImportError:
    from flop_counter import FLOPCounterPython, get_global_counter


def count_model_flops(
    model: Any,
    input_data: Any,
    framework: str = 'auto',
    verbose: bool = True,
    return_details: bool = True
) -> Union[int, Dict[str, Any]]:
    """
    Cuenta FLOPs de un modelo completo con una sola llamada.
    
    Esta función automáticamente detecta el framework, ejecuta el modelo,
    y retorna el conteo de FLOPs con detalles opcionales.
    
    Args:
        model: Modelo a perfilar. Puede ser:
            - torch.nn.Module (PyTorch)
            - tf.keras.Model (TensorFlow/Keras)
            - Callable (función Python)
        
        input_data: Datos de entrada para el modelo. Puede ser:
            - torch.Tensor
            - np.ndarray
            - tf.Tensor
            - tuple/list de cualquiera de los anteriores
        
        framework: Framework a usar. Opciones:
            - 'auto': Detectar automáticamente (default)
            - 'torch': PyTorch
            - 'tensorflow': TensorFlow
            - 'numpy': NumPy/función Python
        
        verbose: Si True, imprime resumen detallado
        
        return_details: Si True, retorna dict con detalles.
                       Si False, retorna solo el número de FLOPs.
    
    Returns:
        Si return_details=True:
            dict con keys:
                - 'total_flops': int
                - 'flops_by_operation': dict
                - 'execution_time_ms': float
                - 'operations': list
                - 'framework': str
                - 'model_info': dict
        
        Si return_details=False:
            int: Total de FLOPs
    
    Examples:
        >>> import torch
        >>> import flop_counter
        >>> 
        >>> # Ejemplo con PyTorch
        >>> model = torch.nn.Linear(784, 10)
        >>> x = torch.randn(32, 784)
        >>> result = count_model_flops(model, x)
        >>> print(f"FLOPs: {result['total_flops']:,}")
        
        >>> # Ejemplo con función Python
        >>> def my_model(x):
        ...     return np.matmul(x, np.random.randn(100, 10))
        >>> x = np.random.randn(32, 100)
        >>> flops = count_model_flops(my_model, x, return_details=False)
        >>> print(f"FLOPs: {flops:,}")
    """
    
    # 1. Detectar framework si es 'auto'
    if framework == 'auto':
        framework = _detect_framework(model, input_data)
    
    # 2. Validar framework
    if framework not in ['torch', 'tensorflow', 'numpy']:
        raise ValueError(f"Framework '{framework}' no soportado. Use: 'torch', 'tensorflow', 'numpy', o 'auto'")
    
    # 3. Obtener información del modelo
    model_info = _get_model_info(model, framework)
    
    # 4. Preparar datos de entrada
    input_data = _prepare_input(input_data, framework)
    
    # 5. Ejecutar modelo y contar FLOPs
    counter = FLOPCounterPython()
    
    import time
    start_time = time.perf_counter()
    
    try:
        with counter.count_flops():
            if framework == 'torch':
                output = _run_torch_model(model, input_data)
            elif framework == 'tensorflow':
                output = _run_tensorflow_model(model, input_data)
            else:  # numpy
                output = _run_numpy_model(model, input_data)
    except Exception as e:
        raise RuntimeError(f"Error ejecutando modelo: {e}")
    
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    
    # 6. Obtener resultados
    total_flops = counter.get_total_flops()
    operations = counter.get_operations()
    
    # 7. Procesar resultados
    if verbose:
        _print_model_summary(
            model_info=model_info,
            total_flops=total_flops,
            operations=operations,
            execution_time_ms=total_time_ms,
            framework=framework
        )
    
    if not return_details:
        return total_flops
    
    # 8. Crear dict con detalles
    flops_by_operation = {}
    for op in operations:
        op_name = op['name']
        flops_by_operation[op_name] = flops_by_operation.get(op_name, 0) + op['flops']
    
    return {
        'total_flops': total_flops,
        'flops_by_operation': flops_by_operation,
        'execution_time_ms': total_time_ms,
        'operations': operations,
        'framework': framework,
        'model_info': model_info,
        'output_shape': _get_shape(output) if hasattr(output, 'shape') else None
    }


def _detect_framework(model: Any, input_data: Any) -> str:
    """Detecta automáticamente el framework del modelo."""
    
    # Detectar PyTorch
    if 'torch' in sys.modules:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'torch'
        if isinstance(input_data, torch.Tensor):
            return 'torch'
    
    # Detectar TensorFlow
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return 'tensorflow'
        if hasattr(tf, 'Tensor') and isinstance(input_data, tf.Tensor):
            return 'tensorflow'
    
    # Default: numpy/función Python
    return 'numpy'


def _get_model_info(model: Any, framework: str) -> Dict[str, Any]:
    """Extrae información del modelo."""
    
    info = {
        'framework': framework,
        'type': type(model).__name__,
        'name': getattr(model, '__name__', type(model).__name__)
    }
    
    if framework == 'torch':
        try:
            import torch
            # Contar parámetros
            if isinstance(model, torch.nn.Module):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                info['total_parameters'] = total_params
                info['trainable_parameters'] = trainable_params
                info['model_name'] = model.__class__.__name__
        except Exception:
            pass
    
    elif framework == 'tensorflow':
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                info['total_parameters'] = model.count_params()
                info['model_name'] = model.name
                info['layers'] = len(model.layers)
        except Exception:
            pass
    
    return info


def _prepare_input(input_data: Any, framework: str) -> Any:
    """Prepara los datos de entrada para el modelo."""
    
    # Si es una tupla o lista, mantenerla como está
    if isinstance(input_data, (tuple, list)):
        return input_data
    
    # Si es un tensor/array, retornarlo directamente
    return input_data


def _get_shape(data: Any) -> Tuple:
    """Obtiene la forma de un tensor/array."""
    if hasattr(data, 'shape'):
        return tuple(data.shape)
    elif isinstance(data, (list, tuple)):
        return (len(data),)
    return None


def _run_torch_model(model: Any, input_data: Any) -> Any:
    """Ejecuta un modelo de PyTorch."""
    import torch
    
    # Poner modelo en modo evaluación
    if isinstance(model, torch.nn.Module):
        model.eval()
    
    # Ejecutar sin gradientes
    with torch.no_grad():
        if isinstance(input_data, (tuple, list)):
            output = model(*input_data)
        else:
            output = model(input_data)
    
    return output


def _run_tensorflow_model(model: Any, input_data: Any) -> Any:
    """Ejecuta un modelo de TensorFlow."""
    import tensorflow as tf
    
    # Ejecutar modelo
    if isinstance(input_data, (tuple, list)):
        output = model(*input_data, training=False)
    else:
        output = model(input_data, training=False)
    
    return output


def _run_numpy_model(model: Callable, input_data: Any) -> Any:
    """Ejecuta un modelo/función Python."""
    
    # Si es callable, ejecutarlo
    if callable(model):
        if isinstance(input_data, (tuple, list)):
            return model(*input_data)
        else:
            return model(input_data)
    
    raise TypeError(f"El modelo debe ser callable, recibido: {type(model)}")


def _print_model_summary(
    model_info: Dict,
    total_flops: int,
    operations: list,
    execution_time_ms: float,
    framework: str
):
    """Imprime un resumen detallado del profiling."""
    
    print("\n" + "="*70)
    print("MODEL PROFILING SUMMARY")
    print("="*70)
    
    # Información del modelo
    print(f"\nModel Information:")
    print(f"  Framework:     {framework.upper()}")
    print(f"  Model Type:    {model_info.get('type', 'Unknown')}")
    print(f"  Model Name:    {model_info.get('name', 'Unknown')}")
    
    if 'total_parameters' in model_info:
        print(f"  Parameters:    {model_info['total_parameters']:,}")
    
    if 'trainable_parameters' in model_info:
        print(f"  Trainable:     {model_info['trainable_parameters']:,}")
    
    # Métricas de FLOPs
    print(f"\nComputational Complexity:")
    print(f"  Total FLOPs:   {total_flops:,}")
    
    # Convertir a unidades más legibles
    if total_flops >= 1e12:
        print(f"                 {total_flops/1e12:.2f} TFLOPs")
    elif total_flops >= 1e9:
        print(f"                 {total_flops/1e9:.2f} GFLOPs")
    elif total_flops >= 1e6:
        print(f"                 {total_flops/1e6:.2f} MFLOPs")
    elif total_flops >= 1e3:
        print(f"                 {total_flops/1e3:.2f} KFLOPs")
    
    print(f"  Execution Time: {execution_time_ms:.2f} ms")
    
    if total_flops > 0 and execution_time_ms > 0:
        flops_per_second = total_flops / (execution_time_ms / 1000)
        print(f"  Throughput:     {flops_per_second/1e9:.2f} GFLOPS/s")
    
    # Breakdown por operación
    if operations:
        print(f"\nOperation Breakdown (Top 10):")
        
        # Agrupar por tipo
        flops_by_op = {}
        count_by_op = {}
        
        for op in operations:
            name = op['name']
            flops_by_op[name] = flops_by_op.get(name, 0) + op['flops']
            count_by_op[name] = count_by_op.get(name, 0) + 1
        
        # Ordenar por FLOPs (descendente)
        sorted_ops = sorted(flops_by_op.items(), key=lambda x: x[1], reverse=True)
        
        print(f"  {'Operation':<20} {'Count':<8} {'FLOPs':<15} {'%':<8}")
        print(f"  {'-'*60}")
        
        for i, (op_name, op_flops) in enumerate(sorted_ops[:10]):
            percentage = (op_flops / total_flops * 100) if total_flops > 0 else 0
            count = count_by_op[op_name]
            print(f"  {op_name:<20} {count:<8} {op_flops:<15,} {percentage:>6.1f}%")
    
    print("\n" + "="*70 + "\n")


# Funciones auxiliares para análisis

def compare_models(
    models: Dict[str, Tuple[Any, Any]],
    framework: str = 'auto',
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Compara múltiples modelos.
    
    Args:
        models: Dict con formato {'nombre': (modelo, input_data)}
        framework: Framework a usar
        verbose: Si True, imprime comparación
    
    Returns:
        Dict con resultados de cada modelo
    
    Example:
        >>> models = {
        ...     'small': (small_model, x),
        ...     'large': (large_model, x)
        ... }
        >>> results = compare_models(models)
    """
    
    results = {}
    
    for name, (model, input_data) in models.items():
        print(f"\nProfiling: {name}")
        print("-" * 50)
        
        result = count_model_flops(
            model=model,
            input_data=input_data,
            framework=framework,
            verbose=False,
            return_details=True
        )
        
        results[name] = result
    
    if verbose:
        _print_comparison(results)
    
    return results


def _print_comparison(results: Dict[str, Dict]):
    """Imprime comparación de modelos."""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<20} {'FLOPs':<15} {'Time (ms)':<12} {'GFLOPS/s':<12}")
    print("-"*70)
    
    for name, result in results.items():
        flops = result['total_flops']
        time_ms = result['execution_time_ms']
        throughput = (flops / (time_ms / 1000)) / 1e9 if time_ms > 0 else 0
        
        print(f"{name:<20} {flops:<15,} {time_ms:<12.2f} {throughput:<12.2f}")
    
    # Encontrar el más eficiente
    if len(results) > 1:
        min_flops = min(r['total_flops'] for r in results.values())
        min_model = [name for name, r in results.items() if r['total_flops'] == min_flops][0]
        
        print(f"\nMost Efficient: {min_model} ({min_flops:,} FLOPs)")
        
        print(f"\nEfficiency Ratios (vs {min_model}):")
        for name, result in results.items():
            if name != min_model:
                ratio = result['total_flops'] / min_flops
                print(f"  {name}: {ratio:.2f}x more FLOPs")
    
    print("\n" + "="*70 + "\n")