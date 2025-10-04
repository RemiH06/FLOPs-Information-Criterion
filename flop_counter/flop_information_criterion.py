"""
FIC: FLOPs Information Criterion (Remi's Information Criterion - RIC)

Este módulo implementa el criterio de información basado en FLOPs
para selección de modelos en machine learning.

Autor: Remi Heredia
Fecha: Septiembre 2025
Versión: 1.0
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Callable, List
import warnings

try:
    from .model_profiler import count_model_flops
    from .flop_counter import FLOPCounterPython
except ImportError:
    from model_profiler import count_model_flops
    from flop_counter import FLOPCounterPython


class FlopInformationCriterion:
    """
    Implementación del FLOPs Information Criterion (FIC/RIC).
    
    El FIC penaliza modelos tanto por su desajuste a los datos como
    por su complejidad computacional medida en FLOPs.
    
    Fórmula general:
        FIC = -2*log(L) + α*log(FLOPs) + β*k
    
    Donde:
        L = verosimilitud del modelo
        FLOPs = operaciones de punto flotante
        k = número de parámetros
        α, β = coeficientes de penalización
    """
    
    def __init__(self, variant: str = 'standard', alpha: Optional[float] = None, 
                 beta: Optional[float] = None, custom_penalty: Optional[Callable] = None,
                 flops_scale: str = 'log_normalized'):
        """
        Inicializa el calculador de FIC.
        
        Args:
            variant: Variante del FIC a usar. Opciones:
                - 'standard' (FIC-S): α=2, β=0
                - 'bic' (FIC-BIC): α crece con n
                - 'hybrid' (FIC-H): α=2, β=1 (RECOMENDADO)
                - 'normalized' (FIC-N): Normalizado por tamaño de muestra
                - 'relative' (FIC-R): Relativo a precisión
                - 'custom': Usar valores α y β proporcionados
            
            alpha: Coeficiente de penalización para FLOPs (solo si variant='custom')
            beta: Coeficiente de penalización para parámetros (solo si variant='custom')
            custom_penalty: Función personalizada para calcular penalización
            flops_scale: Escala para FLOPs. Opciones:
                - 'log': log(FLOPs) [ORIGINAL - poco peso]
                - 'linear_mega': FLOPs / 1e6 [Opción 1]
                - 'log_normalized': log(FLOPs / 1e3) [Opción 2]
                - 'sqrt_mega': sqrt(FLOPs / 1e6) [Opción 3]
                - 'log_params_ratio': log(FLOPs / params) [Tu opción 1]
                - 'params_flops_ratio': log(params / FLOPs) [Tu opción 2]
                - 'log_plus_linear': log(FLOPs) + FLOPs/1e6 [Híbrido]
        """
        self.variant = variant.lower()
        self._validate_variant()
        
        # Configurar coeficientes según variante
        if self.variant == 'custom':
            if alpha is None or beta is None:
                raise ValueError("Para variant='custom', debe especificar alpha y beta")
            self.alpha = alpha
            self.beta = beta
        else:
            self.alpha = alpha  # Se calculará dinámicamente si es necesario
            self.beta = beta
        
        self.custom_penalty = custom_penalty
        self.flops_scale = flops_scale
        
        # Validar escala de FLOPs
        valid_scales = [
            'log', 'linear_mega', 'log_normalized', 'sqrt_mega',
            'log_params_ratio', 'params_flops_ratio', 'log_plus_linear',
            'cube_root_kilo', 'log_flops_per_param', 'parametric_log_linear'
        ]
        if not (self.flops_scale in valid_scales or 
        self.flops_scale.startswith('parametric_log_linear_')):
            raise ValueError(
                f"flops_scale '{self.flops_scale}' no válido. "
                f"Opciones: {', '.join(valid_scales)} o 'parametric_log_linear_X.X'"
            )
        
        # Cache para resultados
        self._cache = {}
    
    def _validate_variant(self):
        """Valida que la variante sea válida."""
        valid_variants = ['standard', 'bic', 'hybrid', 'normalized', 'relative', 'custom']
        if self.variant not in valid_variants:
            raise ValueError(
                f"Variante '{self.variant}' no válida. "
                f"Opciones: {', '.join(valid_variants)}"
            )
    
    def _get_coefficients(self, n: int, accuracy: Optional[float] = None) -> tuple:
        """
        Obtiene los coeficientes α y β según la variante.
        
        Args:
            n: Tamaño de la muestra
            accuracy: Precisión del modelo (para variant='relative')
        
        Returns:
            tuple: (alpha, beta)
        """
        if self.variant == 'standard':
            # FIC-S: similar a AIC
            return (2.0, 0.0)
        
        elif self.variant == 'bic':
            # FIC-BIC: penalización crece con n
            return (np.log(n), 0.0)
        
        elif self.variant == 'hybrid':
            # FIC-H: balance entre FLOPs y parámetros
            return (2.0, 1.0)
        
        elif self.variant == 'normalized':
            # FIC-N: normalizado por n
            return (2.0 / np.log(n) if n > 1 else 2.0, 1.0)
        
        elif self.variant == 'relative':
            # FIC-R: relativo a precisión
            if accuracy is None:
                warnings.warn(
                    "variant='relative' requiere accuracy. Usando variant='standard'.",
                    UserWarning
                )
                return (2.0, 0.0)
            # α escala inversamente con accuracy
            return (2.0 / max(accuracy, 0.01), 0.0)
        
        elif self.variant == 'custom':
            return (self.alpha, self.beta)
        
        else:
            return (2.0, 1.0)  # Default: hybrid
    
    def calculate_log_likelihood(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 task: str = 'regression',
                                 sigma: Optional[float] = None) -> float:
        """
        Calcula -2*log(L) para el modelo.
        
        Args:
            y_true: Valores verdaderos (shape: [n, ...])
            y_pred: Predicciones del modelo (shape: [n, ...])
            task: Tipo de tarea:
                - 'regression': Regresión con errores gaussianos
                - 'classification': Clasificación con cross-entropy
                - 'binary': Clasificación binaria
            sigma: Desviación estándar del error (solo para regresión)
        
        Returns:
            float: -2*log(L)
        """
        n = len(y_true)
        
        if task == 'regression':
            # -2*log(L) para regresión gaussiana
            residuals = y_true - y_pred
            rss = np.sum(residuals ** 2)
            
            if sigma is None:
                # Estimar sigma de los residuos
                sigma = np.sqrt(rss / n)
            
            # Fórmula: n*log(2π) + n*log(σ²) + RSS/σ²
            log_likelihood = n * np.log(2 * np.pi) + n * np.log(sigma**2) + rss / (sigma**2)
            
        elif task == 'classification':
            # -2*log(L) para clasificación multiclase
            # y_pred debe ser probabilidades [n, num_classes]
            # y_true debe ser labels [n] o one-hot [n, num_classes]
            
            # Convertir y_true a índices si es one-hot
            if y_true.ndim > 1:
                y_true_idx = np.argmax(y_true, axis=1)
            else:
                y_true_idx = y_true.astype(int)
            
            # Evitar log(0)
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            
            # Cross-entropy
            log_likelihood = -2 * np.sum(np.log(y_pred_clipped[np.arange(n), y_true_idx]))
            
        elif task == 'binary':
            # -2*log(L) para clasificación binaria
            # y_pred debe ser probabilidades [n]
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            
            log_likelihood = -2 * np.sum(
                y_true * np.log(y_pred_clipped) + 
                (1 - y_true) * np.log(1 - y_pred_clipped)
            )
        
        else:
            raise ValueError(f"task='{task}' no soportado. Use: 'regression', 'classification', 'binary'")
        
        return float(log_likelihood)
    
    def _calculate_flops_penalty(self, flops: int, n_params: int, alpha: float) -> float:
        """
        Calcula la penalización por FLOPs según la escala configurada.
        
        Args:
            flops: Número de FLOPs
            n_params: Número de parámetros (para escalas que usan ratio)
            alpha: Coeficiente α
        
        Returns:
            float: Penalización por FLOPs
        """
        if flops <= 0:
            warnings.warn("FLOPs = 0, penalización por FLOPs es 0", UserWarning)
            return 0.0
        
        if self.flops_scale == 'log':
            # ORIGINAL: log(FLOPs)
            # Problema: Para FLOPs grandes, log es muy pequeño
            penalty = alpha * np.log(flops)
        
        elif self.flops_scale == 'linear_mega':
            # OPCIÓN 1: FLOPs / 1e6 (lineal en millones)
            # Ventaja: Directamente proporcional a FLOPs
            # Problema: Puede crecer demasiado para modelos muy grandes
            penalty = alpha * (flops / 1e6)
        
        elif self.flops_scale == 'log_normalized':
            # OPCIÓN 2: log(FLOPs / 1e3)
            # Ventaja: Reduce el rango antes de aplicar log
            penalty = alpha * np.log(flops / 1e3)
        
        elif self.flops_scale == 'sqrt_mega':
            # OPCIÓN 3: sqrt(FLOPs / 1e6)
            # Ventaja: Más suave que lineal, más fuerte que log
            penalty = alpha * np.sqrt(flops / 1e6)
        
        elif self.flops_scale == 'log_params_ratio':
            # TU OPCIÓN 1: log(FLOPs / params)
            # Ventaja: Normaliza por complejidad del modelo
            if n_params > 0:
                penalty = alpha * np.log(flops / n_params)
            else:
                penalty = alpha * np.log(flops)
        
        elif self.flops_scale == 'params_flops_ratio':
            # TU OPCIÓN 2: log(params / FLOPs)
            # Ventaja: Penaliza modelos con pocos parámetros pero muchos FLOPs
            if n_params > 0:
                penalty = alpha * np.log(n_params / flops)
            else:
                penalty = 0.0
        
        elif self.flops_scale == 'log_plus_linear':
            # HÍBRIDO: log(FLOPs) + FLOPs/1e6
            # Ventaja: Combina crecimiento logarítmico con penalización lineal
            penalty = alpha * (np.log(flops) + flops / 1e6)

        elif self.flops_scale == 'cube_root_kilo':
            # NUEVA: (FLOPs^(1/3)) / 1e3
            # Ventaja: Crece más rápido que log, más lento que sqrt
            # Suaviza diferencias grandes de FLOPs
            penalty = alpha * (flops ** (1/3) / 1e3)

        elif self.flops_scale == 'log_flops_per_param':
            # NUEVA: log(1 + FLOPs/params)
            # Ventaja: Normaliza por parámetros, siempre positivo
            # Penaliza modelos ineficientes (muchos FLOPs por parámetro)
            if n_params > 0:
                penalty = alpha * np.log(1 + flops / n_params)
            else:
                penalty = alpha * np.log(1 + flops)

        elif self.flops_scale == 'parametric_log_linear':
            # PARAMETRIC: α * (λ * log(FLOPs) + (1-λ) * FLOPs/1e6)
            # λ controla el balance entre log y linear
            # Se pasa λ como parte del nombre: 'parametric_log_linear_0.5'
            
            # Extraer λ del nombre de la escala
            parts = self.flops_scale.split('_')
            if len(parts) == 4 and parts[3].replace('.', '').isdigit():
                lambda_val = float(parts[3])
            else:
                lambda_val = 0.5  # default
            
            log_term = np.log(flops) if flops > 0 else 0
            linear_term = flops / 1e6
            penalty = alpha * (lambda_val * log_term + (1 - lambda_val) * linear_term)

        elif self.flops_scale.startswith('parametric_log_linear'):
            # PARAMÉTRICA: α * (λ * log(FLOPs) + (1-λ) * FLOPs/1e6)
            # λ controla balance entre logarítmico y lineal
            # Formato: 'parametric_log_linear_0.5' donde 0.5 es λ
            
            try:
                lambda_val = float(self.flops_scale.split('_')[-1])
                if not (0.0 <= lambda_val <= 1.0):
                    warnings.warn(f"λ={lambda_val} fuera de rango [0,1], usando 0.5", UserWarning)
                    lambda_val = 0.5
            except (ValueError, IndexError):
                warnings.warn("No se pudo extraer λ, usando 0.5", UserWarning)
                lambda_val = 0.5
            
            log_term = np.log(flops) if flops > 0 else 0.0
            linear_term = flops / 1e6
            penalty = alpha * (lambda_val * log_term + (1 - lambda_val) * linear_term)


        else:
            # Default: log
            penalty = alpha * np.log(flops)
        
        return penalty
    
    def calculate_fic(self,
                     log_likelihood: float,
                     flops: int,
                     n_params: int,
                     n_samples: int,
                     accuracy: Optional[float] = None,
                     alpha_override: Optional[float] = None,
                     beta_override: Optional[float] = None) -> Dict[str, float]:
        """
        Calcula el FIC dado los componentes.
        
        Args:
            log_likelihood: -2*log(L) (término de ajuste)
            flops: Número de FLOPs del modelo
            n_params: Número de parámetros del modelo
            n_samples: Tamaño de la muestra
            accuracy: Precisión del modelo (opcional, para variant='relative')
            alpha_override: Sobreescribir α temporalmente para experimentos
            beta_override: Sobreescribir β temporalmente para experimentos
        
        Returns:
            dict con keys:
                - 'fic': Valor del FIC
                - 'log_likelihood_term': Término de verosimilitud
                - 'flops_penalty': Penalización por FLOPs
                - 'params_penalty': Penalización por parámetros
                - 'alpha': Valor de α usado
                - 'beta': Valor de β usado
                - 'flops_scale': Escala de FLOPs usada
        """
        # Obtener coeficientes (permitir override para experimentos)
        if alpha_override is not None and beta_override is not None:
            alpha, beta = alpha_override, beta_override
        else:
            alpha, beta = self._get_coefficients(n_samples, accuracy)
        
        # Calcular términos
        likelihood_term = log_likelihood
        
        # Penalización por FLOPs (usando la escala configurada)
        flops_penalty = self._calculate_flops_penalty(flops, n_params, alpha)
        
        # Penalización por parámetros
        params_penalty = beta * n_params
        
        # Penalización personalizada (si existe)
        if self.custom_penalty is not None:
            custom_term = self.custom_penalty(flops, n_params, n_samples)
        else:
            custom_term = 0.0
        
        # FIC total
        fic_value = likelihood_term + flops_penalty + params_penalty + custom_term
        
        return {
            'fic': fic_value,
            'log_likelihood_term': likelihood_term,
            'flops_penalty': flops_penalty,
            'params_penalty': params_penalty,
            'custom_penalty': custom_term,
            'alpha': alpha,
            'beta': beta,
            'flops_scale': self.flops_scale,
            'variant': self.variant
        }
    
    def evaluate_model(self,
                      model: Any,
                      X: np.ndarray,
                      y_true: np.ndarray,
                      y_pred: Optional[np.ndarray] = None,
                      task: str = 'regression',
                      n_params: Optional[int] = None,
                      framework: str = 'auto',
                      sigma: Optional[float] = None,
                      alpha_override: Optional[float] = None,
                      beta_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Evalúa un modelo completo y calcula su FIC.
        
        Args:
            model: Modelo a evaluar (para contar FLOPs)
            X: Datos de entrada
            y_true: Valores verdaderos
            y_pred: Predicciones (si None, se calcula ejecutando el modelo)
            task: Tipo de tarea ('regression', 'classification', 'binary')
            n_params: Número de parámetros (si None, se intenta inferir)
            framework: Framework del modelo ('auto', 'torch', 'tensorflow', 'numpy')
            sigma: Desviación estándar del error (regresión)
            alpha_override: Override temporal de α para experimentos
            beta_override: Override temporal de β para experimentos
        
        Returns:
            dict con resultados completos del FIC y métricas adicionales
        """
        n_samples = len(y_true)
        
        # 1. Contar FLOPs
        flop_result = count_model_flops(
            model=model,
            input_data=X,
            framework=framework,
            verbose=False,
            return_details=True
        )
        
        flops = flop_result['total_flops']
        
        # 2. Inferir número de parámetros si no se proporciona
        if n_params is None:
            if 'model_info' in flop_result and 'total_parameters' in flop_result['model_info']:
                n_params = flop_result['model_info']['total_parameters']
            else:
                # Estimación: contar parámetros del modelo
                n_params = self._estimate_parameters(model, framework)
        
        # 3. Obtener predicciones si no se proporcionan
        if y_pred is None:
            y_pred = self._get_predictions(model, X, framework, task)
        
        # 4. Calcular log-verosimilitud
        log_likelihood = self.calculate_log_likelihood(y_true, y_pred, task, sigma)
        
        # 5. Calcular métricas adicionales
        accuracy = self._calculate_accuracy(y_true, y_pred, task)
        
        # 6. Calcular FIC (con posibles overrides)
        fic_result = self.calculate_fic(
            log_likelihood=log_likelihood,
            flops=flops,
            n_params=n_params,
            n_samples=n_samples,
            accuracy=accuracy,
            alpha_override=alpha_override,
            beta_override=beta_override
        )
        
        # 7. Combinar resultados
        result = {
            **fic_result,
            'n_samples': n_samples,
            'n_params': n_params,
            'flops': flops,
            'accuracy': accuracy,
            'model_info': flop_result.get('model_info', {}),
            'execution_time_ms': flop_result.get('execution_time_ms', 0.0)
        }
        
        return result
    
    def compare_models(self,
                      models: Dict[str, tuple],
                      task: str = 'regression',
                      framework: str = 'auto') -> Dict[str, Dict]:
        """
        Compara múltiples modelos usando FIC.
        
        Args:
            models: Dict con formato:
                {'nombre': (model, X, y_true, y_pred), ...}
                donde y_pred es opcional
            task: Tipo de tarea
            framework: Framework de los modelos
        
        Returns:
            dict con resultados de cada modelo, ordenados por FIC (menor primero)
        """
        results = {}
        
        for name, model_data in models.items():
            if len(model_data) == 3:
                model, X, y_true = model_data
                y_pred = None
            elif len(model_data) == 4:
                model, X, y_true, y_pred = model_data
            else:
                raise ValueError(f"Formato incorrecto para modelo '{name}'")
            
            # Evaluar modelo
            result = self.evaluate_model(
                model=model,
                X=X,
                y_true=y_true,
                y_pred=y_pred,
                task=task,
                framework=framework
            )
            
            results[name] = result
        
        # Ordenar por FIC (menor es mejor)
        results = dict(sorted(results.items(), key=lambda x: x[1]['fic']))
        
        # Calcular diferencias relativas
        fic_values = [r['fic'] for r in results.values()]
        min_fic = min(fic_values)
        
        for name, result in results.items():
            result['delta_fic'] = result['fic'] - min_fic
            result['is_best'] = (result['delta_fic'] < 0.01)  # Tolerancia numérica
        
        return results
    
    def _estimate_parameters(self, model: Any, framework: str) -> int:
        """Estima el número de parámetros del modelo."""
        try:
            if framework == 'torch' or 'torch' in str(type(model)):
                import torch
                if isinstance(model, torch.nn.Module):
                    return sum(p.numel() for p in model.parameters())
            
            elif framework == 'tensorflow' or 'tensorflow' in str(type(model)):
                import tensorflow as tf
                if isinstance(model, tf.keras.Model):
                    return model.count_params()
            
            # Si no se puede inferir, usar una estimación conservadora
            warnings.warn(
                "No se pudo inferir el número de parámetros. Usando 0.",
                UserWarning
            )
            return 0
        
        except Exception:
            return 0
    
    def _get_predictions(self, model: Any, X: np.ndarray, 
                        framework: str, task: str) -> np.ndarray:
        """Obtiene predicciones del modelo."""
        if framework == 'torch' or 'torch' in str(type(model)):
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_pred = model(X_tensor).numpy()
        
        elif framework == 'tensorflow' or 'tensorflow' in str(type(model)):
            y_pred = model(X, training=False).numpy()
        
        elif callable(model):
            y_pred = model(X)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
        
        else:
            raise TypeError(f"No se puede obtener predicciones de modelo tipo {type(model)}")
        
        return y_pred
    
    def _calculate_accuracy(self, y_true: np.ndarray, 
                           y_pred: np.ndarray, task: str) -> float:
        """Calcula la precisión del modelo."""
        if task == 'regression':
            # R² score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return max(0.0, r2)  # R² puede ser negativo
        
        elif task == 'classification':
            # Accuracy
            if y_pred.ndim > 1:
                y_pred_labels = np.argmax(y_pred, axis=1)
            else:
                y_pred_labels = y_pred
            
            if y_true.ndim > 1:
                y_true_labels = np.argmax(y_true, axis=1)
            else:
                y_true_labels = y_true
            
            return np.mean(y_pred_labels == y_true_labels)
        
        elif task == 'binary':
            # Binary accuracy
            y_pred_binary = (y_pred > 0.5).astype(int)
            return np.mean(y_pred_binary == y_true)
        
        else:
            return 0.0


# Funciones de conveniencia

def calculate_fic(log_likelihood: float,
                 flops: int,
                 n_params: int,
                 n_samples: int,
                 variant: str = 'hybrid',
                 alpha: Optional[float] = None,
                 beta: Optional[float] = None) -> float:
    """
    Función de conveniencia para calcular FIC directamente.
    
    Args:
        log_likelihood: -2*log(L)
        flops: Número de FLOPs
        n_params: Número de parámetros
        n_samples: Tamaño de muestra
        variant: Variante del FIC
        alpha: Coeficiente α (para variant='custom')
        beta: Coeficiente β (para variant='custom')
    
    Returns:
        float: Valor del FIC
    """
    fic_calc = FlopInformationCriterion(variant=variant, alpha=alpha, beta=beta)
    result = fic_calc.calculate_fic(log_likelihood, flops, n_params, n_samples)
    return result['fic']


def evaluate_model_fic(model: Any,
                       X: np.ndarray,
                       y_true: np.ndarray,
                       task: str = 'regression',
                       variant: str = 'hybrid',
                       framework: str = 'auto') -> Dict[str, Any]:
    """
    Función de conveniencia para evaluar un modelo con FIC.
    
    Args:
        model: Modelo a evaluar
        X: Datos de entrada
        y_true: Valores verdaderos
        task: Tipo de tarea
        variant: Variante del FIC
        framework: Framework del modelo
    
    Returns:
        dict con resultados del FIC
    """
    fic_calc = FlopInformationCriterion(variant=variant)
    return fic_calc.evaluate_model(model, X, y_true, task=task, framework=framework)


def compare_models_fic(models: Dict[str, tuple],
                       task: str = 'regression',
                       variant: str = 'hybrid',
                       framework: str = 'auto') -> Dict[str, Dict]:
    """
    Función de conveniencia para comparar múltiples modelos con FIC.
    
    Args:
        models: Dict con modelos a comparar
        task: Tipo de tarea
        variant: Variante del FIC
        framework: Framework de los modelos
    
    Returns:
        dict con comparación de modelos ordenados por FIC
    """
    fic_calc = FlopInformationCriterion(variant=variant)
    return fic_calc.compare_models(models, task=task, framework=framework)