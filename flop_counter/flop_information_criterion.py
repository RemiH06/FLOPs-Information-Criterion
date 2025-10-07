"""
FIC: FLOPs Information Criterion (Remi's Information Criterion - RIC)

Este módulo implementa el criterio de información basado en FLOPs
para selección de modelos en machine learning.

Autor: Remi Heredia
Fecha: Octubre 2025
Versión: 2.0 (Limpia)
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Callable
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
    
    Fórmula paramétrica (recomendada):
        FIC = -2*log(L) + α*[λ*log(FLOPs) + (1-λ)*FLOPs/10⁶] + β*k
    
    Donde:
        L = verosimilitud del modelo
        FLOPs = operaciones de punto flotante
        k = número de parámetros
        α = coeficiente de penalización de FLOPs (recomendado: 5.0)
        β = coeficiente de penalización de parámetros (recomendado: 0.5)
        λ = balance log/lineal (recomendado: 0.3)
    
    Configuraciones recomendadas:
        - Móvil/Edge: α=7.0, β=0.3, λ=0.2 (penaliza fuerte FLOPs)
        - Balanceado: α=5.0, β=0.5, λ=0.3 (default)
        - Servidor: α=2.0, β=1.0, λ=0.5 (menos restrictivo)
    """
    
    def __init__(self, variant: str = 'custom', alpha: float = 5.0, 
                 beta: float = 0.5, lambda_balance: float = 0.3,
                 custom_penalty: Optional[Callable] = None):
        """
        Inicializa el calculador de FIC.
        
        Args:
            variant: Variante del FIC:
                - 'custom': Usa α, β, λ proporcionados (RECOMENDADO)
                - 'standard': α=2, β=0 (similar a AIC tradicional)
                - 'hybrid': α=2, β=1 (balance clásico)
            
            alpha: Coeficiente de penalización para FLOPs
                   Mayor valor = penaliza más FLOPs (recomendado: 5.0)
            
            beta: Coeficiente de penalización para parámetros
                  Mayor valor = penaliza más parámetros (recomendado: 0.5)
            
            lambda_balance: Balance entre término log y lineal de FLOPs (0 a 1)
                           λ=0: 100% lineal (penaliza diferencias absolutas)
                           λ=1: 100% logarítmico (penaliza diferencias relativas)
                           λ=0.3: 30% log, 70% lineal (recomendado)
            
            custom_penalty: Función personalizada adicional (opcional)
        """
        self.variant = variant.lower()
        self._validate_variant()
        
        # Configurar coeficientes según variante
        if self.variant == 'custom':
            self.alpha = alpha
            self.beta = beta
            self.lambda_balance = lambda_balance
        elif self.variant == 'standard':
            self.alpha = 2.0
            self.beta = 0.0
            self.lambda_balance = 0.5
        elif self.variant == 'hybrid':
            self.alpha = 2.0
            self.beta = 1.0
            self.lambda_balance = 0.5
        else:
            # Fallback a custom con valores default
            self.alpha = alpha
            self.beta = beta
            self.lambda_balance = lambda_balance
        
        self.custom_penalty = custom_penalty
        
        # Validar parámetros
        if not (0.0 <= self.lambda_balance <= 1.0):
            warnings.warn(
                f"lambda_balance={self.lambda_balance} fuera de rango [0,1]. "
                f"Usando 0.3",
                UserWarning
            )
            self.lambda_balance = 0.3
        
        # Cache para resultados
        self._cache = {}
    
    def _validate_variant(self):
        """Valida que la variante sea válida."""
        valid_variants = ['standard', 'hybrid', 'custom']
        if self.variant not in valid_variants:
            raise ValueError(
                f"Variante '{self.variant}' no válida. "
                f"Opciones: {', '.join(valid_variants)}"
            )
    
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
                - 'classification': Clasificación multiclase con cross-entropy
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
    
    def _calculate_flops_penalty(self, flops: int, alpha: float, lambda_balance: float) -> float:
        """
        Calcula la penalización por FLOPs usando escala paramétrica.
        
        Fórmula: α * [λ * log(FLOPs) + (1-λ) * FLOPs/10⁶]
        
        Args:
            flops: Número de FLOPs
            alpha: Coeficiente α
            lambda_balance: Balance λ entre log y lineal
        
        Returns:
            float: Penalización por FLOPs
        """
        if flops <= 0:
            warnings.warn("FLOPs = 0, penalización por FLOPs es 0", UserWarning)
            return 0.0
        
        # Término logarítmico: sensible a diferencias relativas
        log_term = np.log(flops)
        
        # Término lineal: sensible a diferencias absolutas
        linear_term = flops / 1e6
        
        # Combinación balanceada por λ
        penalty = alpha * (lambda_balance * log_term + (1 - lambda_balance) * linear_term)
        
        return penalty
    
    def calculate_fic(self,
                     log_likelihood: float,
                     flops: int,
                     n_params: int,
                     n_samples: int,
                     accuracy: Optional[float] = None) -> Dict[str, float]:
        """
        Calcula el FIC dado los componentes.
        
        Args:
            log_likelihood: -2*log(L) (término de ajuste)
            flops: Número de FLOPs del modelo
            n_params: Número de parámetros del modelo
            n_samples: Tamaño de la muestra
            accuracy: Precisión del modelo (opcional, para información)
        
        Returns:
            dict con keys:
                - 'fic': Valor del FIC
                - 'log_likelihood_term': Término de verosimilitud
                - 'flops_penalty': Penalización por FLOPs
                - 'params_penalty': Penalización por parámetros
                - 'alpha': Valor de α usado
                - 'beta': Valor de β usado
                - 'lambda': Valor de λ usado
        """
        # Calcular términos
        likelihood_term = log_likelihood
        
        # Penalización por FLOPs
        flops_penalty = self._calculate_flops_penalty(flops, self.alpha, self.lambda_balance)
        
        # Penalización por parámetros
        params_penalty = self.beta * n_params
        
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
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda': self.lambda_balance,
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
                      sigma: Optional[float] = None) -> Dict[str, Any]:
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
                n_params = self._estimate_parameters(model, framework)
        
        # 3. Obtener predicciones si no se proporcionan
        if y_pred is None:
            y_pred = self._get_predictions(model, X, framework, task)
        
        # 4. Calcular log-verosimilitud
        log_likelihood = self.calculate_log_likelihood(y_true, y_pred, task, sigma)
        
        # 5. Calcular métricas adicionales
        accuracy = self._calculate_accuracy(y_true, y_pred, task)
        
        # 6. Calcular FIC
        fic_result = self.calculate_fic(
            log_likelihood=log_likelihood,
            flops=flops,
            n_params=n_params,
            n_samples=n_samples,
            accuracy=accuracy
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
            result['is_best'] = (result['delta_fic'] < 0.01)
        
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
                if isinstance(X, torch.Tensor):
                    X_tensor = X
                else:
                    X_tensor = torch.FloatTensor(X)
                
                y_pred = model(X_tensor)
                
                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.cpu().numpy()
        
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
            return max(0.0, r2)
        
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


# ============================================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================================

def calculate_fic(log_likelihood: float,
                 flops: int,
                 n_params: int,
                 n_samples: int,
                 alpha: float = 5.0,
                 beta: float = 0.5,
                 lambda_balance: float = 0.3) -> float:
    """
    Función de conveniencia para calcular FIC directamente.
    
    Args:
        log_likelihood: -2*log(L)
        flops: Número de FLOPs
        n_params: Número de parámetros
        n_samples: Tamaño de muestra
        alpha: Coeficiente α (default: 5.0)
        beta: Coeficiente β (default: 0.5)
        lambda_balance: Balance λ (default: 0.3)
    
    Returns:
        float: Valor del FIC
    """
    fic_calc = FlopInformationCriterion(
        variant='custom',
        alpha=alpha,
        beta=beta,
        lambda_balance=lambda_balance
    )
    result = fic_calc.calculate_fic(log_likelihood, flops, n_params, n_samples)
    return result['fic']


def evaluate_model_fic(model: Any,
                       X: np.ndarray,
                       y_true: np.ndarray,
                       task: str = 'regression',
                       alpha: float = 5.0,
                       beta: float = 0.5,
                       lambda_balance: float = 0.3,
                       framework: str = 'auto') -> Dict[str, Any]:
    """
    Función de conveniencia para evaluar un modelo con FIC.
    
    Args:
        model: Modelo a evaluar
        X: Datos de entrada
        y_true: Valores verdaderos
        task: Tipo de tarea
        alpha: Coeficiente α (default: 5.0)
        beta: Coeficiente β (default: 0.5)
        lambda_balance: Balance λ (default: 0.3)
        framework: Framework del modelo
    
    Returns:
        dict con resultados del FIC
    """
    fic_calc = FlopInformationCriterion(
        variant='custom',
        alpha=alpha,
        beta=beta,
        lambda_balance=lambda_balance
    )
    return fic_calc.evaluate_model(model, X, y_true, task=task, framework=framework)


def compare_models_fic(models: Dict[str, tuple],
                       task: str = 'regression',
                       alpha: float = 5.0,
                       beta: float = 0.5,
                       lambda_balance: float = 0.3,
                       framework: str = 'auto') -> Dict[str, Dict]:
    """
    Función de conveniencia para comparar múltiples modelos con FIC.
    
    Args:
        models: Dict con modelos a comparar
        task: Tipo de tarea
        alpha: Coeficiente α (default: 5.0)
        beta: Coeficiente β (default: 0.5)
        lambda_balance: Balance λ (default: 0.3)
        framework: Framework de los modelos
    
    Returns:
        dict con comparación de modelos ordenados por FIC
    """
    fic_calc = FlopInformationCriterion(
        variant='custom',
        alpha=alpha,
        beta=beta,
        lambda_balance=lambda_balance
    )
    return fic_calc.compare_models(models, task=task, framework=framework)