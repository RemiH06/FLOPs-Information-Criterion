#!/usr/bin/env python3
"""
Ejemplos de uso del FIC (FLOPs Information Criterion)
Comparación con criterios tradicionales: AIC, BIC, y diferentes escalas de FLOPs
"""

import sys
import os

# Fix para OpenMP warning en Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Añadir el directorio padre
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import flop_counter
from flop_counter.flop_information_criterion import FlopInformationCriterion

# Configurar seed para reproducibilidad
np.random.seed(42)

print("="*80)
print("FIC: FLOPs INFORMATION CRITERION - EJEMPLOS DE USO")
print("="*80)

# ============================================================================
# FUNCIONES AUXILIARES: CRITERIOS TRADICIONALES
# ============================================================================

def calculate_aic(log_likelihood: float, k: int) -> float:
    """AIC: Akaike Information Criterion
    AIC = -2*log(L) + 2*k"""
    return log_likelihood + 2 * k


def calculate_bic(log_likelihood: float, k: int, n: int) -> float:
    """BIC: Bayesian Information Criterion
    BIC = -2*log(L) + k*log(n)"""
    return log_likelihood + k * np.log(n)


def calculate_hqic(log_likelihood: float, k: int, n: int) -> float:
    """HQIC: Hannan-Quinn Information Criterion
    HQIC = -2*log(L) + 2*k*log(log(n))"""
    return log_likelihood + 2 * k * np.log(np.log(n))


def print_criteria_comparison(name: str, results: dict, show_all: bool = False):
    """Imprime comparación de criterios de información."""
    
    print(f"\n{'='*80}")
    print(f"MODELO: {name}")
    print(f"{'='*80}")
    
    # Información básica
    print(f"\nInformación del Modelo:")
    print(f"  Parámetros:       {results.get('n_params', 0):,}")
    print(f"  FLOPs:            {results.get('flops', 0):,}")
    print(f"  Muestras:         {results.get('n_samples', 0)}")
    print(f"  Log-Likelihood:   {results.get('log_likelihood_term', 0):.2f}")
    
    if 'accuracy' in results:
        acc_metric = 'R²' if results.get('accuracy', 0) <= 1 else 'Accuracy'
        print(f"  {acc_metric}:            {results['accuracy']:.4f}")
    
    # Criterios de información
    print(f"\nCriterios de Información:")
    print(f"  {'Criterio':<15} {'Valor':<15} {'Penalización':<25}")
    print(f"  {'-'*55}")
    
    # AIC
    if 'aic' in results:
        print(f"  {'AIC':<15} {results['aic']:<15.2f} {'2k':<25}")
    
    # BIC
    if 'bic' in results:
        print(f"  {'BIC':<15} {results['bic']:<15.2f} {'k*log(n)':<25}")
    
    # FIC (destacado)
    flops_scale = results.get('flops_scale', 'log')
    alpha = results.get('alpha', 2.0)
    beta = results.get('beta', 1.0)
    penalty_formula = f"α*{flops_scale}(FLOPs) + β*k"
    print(f"  {'FIC (RIC)':<15} {results['fic']:<15.2f} {penalty_formula:<25} ⭐")
    
    # Criterios adicionales
    if show_all:
        if 'hqic' in results:
            print(f"  {'HQIC':<15} {results['hqic']:<15.2f} {'2k*log(log(n))':<25}")
    
    # Desglose del FIC
    print(f"\nDesglose del FIC:")
    print(f"  Ajuste (likelihood):       {results['log_likelihood_term']:.2f}")
    print(f"  Penalización FLOPs:        {results['flops_penalty']:.2f}")
    print(f"  Penalización parámetros:   {results['params_penalty']:.2f}")
    print(f"  Escala FLOPs: {flops_scale}, α={alpha:.2f}, β={beta:.2f}")
    print(f"  Total FIC:                 {results['fic']:.2f}")


def compare_all_criteria(models_results: dict, show_all: bool = False):
    """Compara todos los modelos según diferentes criterios."""
    
    print(f"\n{'='*80}")
    print("COMPARACIÓN DE MODELOS")
    print(f"{'='*80}")
    
    model_names = list(models_results.keys())
    criteria = ['AIC', 'BIC', 'FIC']
    
    if show_all:
        criteria.extend(['HQIC'])
    
    # Tabla comparativa
    print(f"\n{'Modelo':<20}", end="")
    for criterion in criteria:
        print(f"{criterion:>12}", end="")
    print()
    print("-" * (20 + 12 * len(criteria)))
    
    for name in model_names:
        res = models_results[name]
        print(f"{name:<20}", end="")
        
        for criterion in criteria:
            key = criterion.lower()
            value = res.get(key, float('nan'))
            print(f"{value:>12.2f}", end="")
        print()
    
    # Mejores modelos según cada criterio
    print(f"\n{'Criterio':<15} {'Mejor Modelo':<20} {'Valor':<15}")
    print("-" * 50)
    
    for criterion in criteria:
        key = criterion.lower()
        best_name = min(model_names, key=lambda x: models_results[x].get(key, float('inf')))
        best_value = models_results[best_name][key]
        print(f"{criterion:<15} {best_name:<20} {best_value:<15.2f}")
    
    # Diferencias ΔFIC
    print(f"\n{'='*80}")
    print("ΔFIC: Diferencias respecto al mejor modelo según FIC")
    print(f"{'='*80}")
    
    fic_values = {name: res['fic'] for name, res in models_results.items()}
    best_fic = min(fic_values.values())
    
    print(f"\n{'Modelo':<20} {'FIC':<15} {'ΔFIC':<15} {'Interpretación':<30}")
    print("-" * 80)
    
    for name in sorted(model_names, key=lambda x: fic_values[x]):
        fic = fic_values[name]
        delta_fic = fic - best_fic
        
        if delta_fic < 2:
            interpretation = "Equivalente al mejor"
        elif delta_fic < 10:
            interpretation = "Evidencia sustancial contra"
        else:
            interpretation = "Evidencia fuerte contra"
        
        marker = "⭐ MEJOR" if delta_fic < 0.01 else ""
        
        print(f"{name:<20} {fic:<15.2f} {delta_fic:<15.2f} {interpretation:<30} {marker}")


# ============================================================================
# EJEMPLO 1: Regresión Lineal - Impacto de Escalas de FLOPs
# ============================================================================

print(f"\n{'='*80}")
print("EJEMPLO 1: REGRESIÓN LINEAL - COMPARACIÓN DE ESCALAS DE FLOPs")
print(f"{'='*80}")

# Generar datos sintéticos
n_samples = 100
X_train = np.random.randn(n_samples, 1)
true_slope = 2.5
true_intercept = 1.0
noise = np.random.randn(n_samples) * 0.5
y_train = true_slope * X_train.squeeze() + true_intercept + noise

def linear_model_simple(X):
    """Modelo lineal simple: y = ax + b"""
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    return X_extended @ W

def linear_model_complex(X):
    """Modelo lineal con operaciones innecesarias (más FLOPs)"""
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    
    # Operaciones adicionales costosas
    temp = X_extended @ W
    # Identidad costosa: exp(log(x)) = x
    temp = np.exp(np.log(np.abs(temp) + 1e-10)) * np.sign(temp)
    # Multiplicación innecesaria
    temp = temp @ np.eye(1)
    # Más operaciones
    temp = np.sin(np.arcsin(np.clip(temp / (np.max(np.abs(temp)) + 1e-10), -0.99, 0.99))) * (np.max(np.abs(temp)) + 1e-10)
    
    return temp

print("\n" + "="*80)
print("COMPARANDO DIFERENTES ESCALAS DE FLOPs")
print("="*80)

# Probar diferentes escalas
scales = ['log', 'linear_mega', 'log_normalized', 'sqrt_mega', 'log_plus_linear']

for scale in scales:
    print(f"\n{'='*80}")
    print(f"ESCALA: {scale}")
    print(f"{'='*80}")
    
    fic_calc = FlopInformationCriterion(variant='hybrid', flops_scale=scale)
    
    result_simple = fic_calc.evaluate_model(
        model=linear_model_simple,
        X=X_train,
        y_true=y_train,
        task='regression',
        n_params=2,
        framework='numpy'
    )
    
    result_complex = fic_calc.evaluate_model(
        model=linear_model_complex,
        X=X_train,
        y_true=y_train,
        task='regression',
        n_params=2,
        framework='numpy'
    )
    
    # Agregar criterios tradicionales
    for result in [result_simple, result_complex]:
        result['aic'] = calculate_aic(result['log_likelihood_term'], result['n_params'])
        result['bic'] = calculate_bic(result['log_likelihood_term'], result['n_params'], result['n_samples'])
    
    models = {
        'Simple': result_simple,
        'Complejo': result_complex
    }
    
    compare_all_criteria(models, show_all=False)
    
    # Análisis de diferencia
    diff_flops = result_complex['flops'] - result_simple['flops']
    diff_penalty = result_complex['flops_penalty'] - result_simple['flops_penalty']
    diff_fic = result_complex['fic'] - result_simple['fic']
    
    print(f"\n📊 Análisis de la escala {scale}:")
    print(f"   Diferencia FLOPs:        {diff_flops:,}")
    print(f"   Diferencia Penalización: {diff_penalty:.2f}")
    print(f"   Diferencia FIC total:    {diff_fic:.2f}")
    
    if diff_fic > 1:
        print(f"   ✅ FIC detecta correctamente que el modelo complejo es peor")
    else:
        print(f"   ⚠️  FIC no distingue bien los modelos con esta escala")

# ============================================================================
# EJEMPLO 2: Redes Neuronales - Arquitecturas Diferentes
# ============================================================================

print(f"\n\n{'='*80}")
print("EJEMPLO 2: REDES NEURONALES - COMPARACIÓN DE ARQUITECTURAS")
print(f"{'='*80}")

# Datos de clasificación
n_samples = 200
n_features = 20
n_classes = 3

X_train = np.random.randn(n_samples, n_features)
y_train = np.random.randint(0, n_classes, n_samples)

def softmax(x):
    """Softmax estable"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def wide_shallow_net(X):
    """Red ancha y poco profunda: 20 -> 100 -> 3"""
    W1 = np.random.randn(20, 100) * 0.01
    b1 = np.zeros(100)
    W2 = np.random.randn(100, 3) * 0.01
    b2 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)  # ReLU
    logits = h1 @ W2 + b2
    return softmax(logits)

def deep_narrow_net(X):
    """Red profunda y estrecha: 20 -> 30 -> 30 -> 30 -> 3"""
    W1 = np.random.randn(20, 30) * 0.01
    b1 = np.zeros(30)
    W2 = np.random.randn(30, 30) * 0.01
    b2 = np.zeros(30)
    W3 = np.random.randn(30, 30) * 0.01
    b3 = np.zeros(30)
    W4 = np.random.randn(30, 3) * 0.01
    b4 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    h3 = np.maximum(0, h2 @ W3 + b3)
    logits = h3 @ W4 + b4
    return softmax(logits)

def balanced_net(X):
    """Red balanceada: 20 -> 50 -> 25 -> 3"""
    W1 = np.random.randn(20, 50) * 0.01
    b1 = np.zeros(50)
    W2 = np.random.randn(50, 25) * 0.01
    b2 = np.zeros(25)
    W3 = np.random.randn(25, 3) * 0.01
    b3 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    return softmax(logits)

# Parámetros de cada modelo
n_params_wide = (20*100 + 100) + (100*3 + 3)  # 2303
n_params_deep = (20*30 + 30) + (30*30 + 30) + (30*30 + 30) + (30*3 + 3)  # 2523
n_params_balanced = (20*50 + 50) + (50*25 + 25) + (25*3 + 3)  # 2353

print("\nEvaluando redes neuronales con FIC-Hybrid (escala log_normalized)...")

fic_calc = FlopInformationCriterion(variant='hybrid', flops_scale='log_normalized')

results_nn = {}

for name, model, n_params in [
    ("Wide-Shallow", wide_shallow_net, n_params_wide),
    ("Deep-Narrow", deep_narrow_net, n_params_deep),
    ("Balanced", balanced_net, n_params_balanced)
]:
    print(f"  Evaluando: {name}...")
    
    result = fic_calc.evaluate_model(
        model=model,
        X=X_train,
        y_true=y_train,
        task='classification',
        n_params=n_params,
        framework='numpy'
    )
    
    # Criterios tradicionales
    result['aic'] = calculate_aic(result['log_likelihood_term'], n_params)
    result['bic'] = calculate_bic(result['log_likelihood_term'], n_params, n_samples)
    result['hqic'] = calculate_hqic(result['log_likelihood_term'], n_params, n_samples)
    
    results_nn[name] = result
    
    print_criteria_comparison(name, result, show_all=False)

compare_all_criteria(results_nn, show_all=True)

print("\n💡 Observaciones:")
print("   - AIC/BIC penalizan solo por # de parámetros")
print("   - FIC captura la diferencia en profundidad (más capas = más FLOPs)")
print("   - Deep-Narrow tiene más FLOPs que Wide-Shallow a pesar de parámetros similares")

# ============================================================================
# EJEMPLO 3: Comparación de Variantes del FIC
# ============================================================================

print(f"\n\n{'='*80}")
print("EJEMPLO 3: COMPARACIÓN DE VARIANTES DEL FIC")
print(f"{'='*80}")

variants = ['standard', 'bic', 'hybrid', 'normalized']

print("\nComparando variantes con el modelo Wide-Shallow...")

for variant in variants:
    print(f"\n{'='*80}")
    print(f"VARIANTE: {variant.upper()}")
    print(f"{'='*80}")
    
    fic_calc = FlopInformationCriterion(variant=variant, flops_scale='log_normalized')
    
    result = fic_calc.evaluate_model(
        model=wide_shallow_net,
        X=X_train,
        y_true=y_train,
        task='classification',
        n_params=n_params_wide,
        framework='numpy'
    )
    
    print(f"\nResultados para variante '{variant}':")
    print(f"  FIC:                      {result['fic']:.2f}")
    print(f"  Log-Likelihood:           {result['log_likelihood_term']:.2f}")
    print(f"  Penalización FLOPs:       {result['flops_penalty']:.2f}")
    print(f"  Penalización parámetros:  {result['params_penalty']:.2f}")
    print(f"  Coeficientes: α={result['alpha']:.2f}, β={result['beta']:.2f}")

print("\n💡 Explicación de variantes:")
print("   - standard (FIC-S): α=2, β=0 (solo FLOPs, similar a AIC)")
print("   - bic (FIC-BIC): α=log(n), β=0 (penaliza más con más datos)")
print("   - hybrid (FIC-H): α=2, β=1 (balance FLOPs + parámetros) [RECOMENDADO]")
print("   - normalized (FIC-N): α ajustado por tamaño de muestra")

# ============================================================================
# EJEMPLO 4: Escalas de FLOPs basadas en ratios
# ============================================================================

print(f"\n\n{'='*80}")
print("EJEMPLO 4: ESCALAS DE FLOPs BASADAS EN RATIOS")
print(f"{'='*80}")

ratio_scales = ['log_params_ratio', 'params_flops_ratio']

for scale in ratio_scales:
    print(f"\n{'='*80}")
    print(f"ESCALA: {scale}")
    print(f"{'='*80}")
    
    fic_calc = FlopInformationCriterion(variant='hybrid', flops_scale=scale)
    
    results_ratio = {}
    
    for name, model, n_params in [
        ("Wide-Shallow", wide_shallow_net, n_params_wide),
        ("Deep-Narrow", deep_narrow_net, n_params_deep),
    ]:
        result = fic_calc.evaluate_model(
            model=model,
            X=X_train,
            y_true=y_train,
            task='classification',
            n_params=n_params,
            framework='numpy'
        )
        
        result['aic'] = calculate_aic(result['log_likelihood_term'], n_params)
        result['bic'] = calculate_bic(result['log_likelihood_term'], n_params, n_samples)
        
        results_ratio[name] = result
    
    compare_all_criteria(results_ratio, show_all=False)
    
    print(f"\n📊 Análisis de {scale}:")
    if scale == 'log_params_ratio':
        print("   Esta escala normaliza FLOPs por # de parámetros")
        print("   Útil cuando quieres penalizar modelos ineficientes")
        print("   (muchos FLOPs por parámetro)")
    else:
        print("   Esta escala es inversa: penaliza modelos con pocos parámetros")
        print("   pero muchos FLOPs (ej: arquitecturas con operaciones repetitivas)")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print(f"\n\n{'='*80}")
print("RESUMEN: GUÍA DE USO DEL FIC (RIC)")
print(f"{'='*80}")

print("""
✅ VENTAJAS DEL FIC:

1. Captura complejidad computacional real
   - AIC/BIC solo consideran # de parámetros
   - FIC considera FLOPs (costo de ejecución)

2. Distingue modelos con igual # de parámetros
   - Dos redes con 1000 parámetros pueden tener FLOPs muy diferentes
   - FIC penaliza apropiadamente el modelo más costoso

3. Refleja la arquitectura del modelo
   - Profundidad, ancho, conexiones skip
   - Todo se traduce en FLOPs

4. Útil para deployment
   - Selecciona modelos eficientes para producción
   - Balance entre precisión y costo computacional

📊 ESCALAS DE FLOPs RECOMENDADAS:

- 'log_normalized': log(FLOPs/1e3) [RECOMENDADO GENERAL]
  → Balancea bien entre modelos pequeños y grandes

- 'sqrt_mega': sqrt(FLOPs/1e6) [PARA MODELOS GRANDES]
  → Más suave que lineal, detecta bien diferencias grandes

- 'log_plus_linear': log(FLOPs) + FLOPs/1e6 [HÍBRIDO]
  → Combina crecimiento logarítmico con penalización lineal

- 'linear_mega': FLOPs/1e6 [PARA DEPLOYMENT]
  → Penaliza fuertemente FLOPs (útil en móviles/edge)

🎯 VARIANTES RECOMENDADAS:

- FIC-Hybrid (α=2, β=1): Balance entre FLOPs y parámetros [DEFAULT]
- FIC-Standard (α=2, β=0): Solo FLOPs, ignora # de parámetros
- FIC-BIC (α=log(n), β=0): Penaliza más con datasets grandes

💡 CUÁNDO USAR CADA CRITERIO:

- AIC:  Selección tradicional, muestra pequeña
- BIC:  Preferencia por modelos simples (penaliza más)
- FIC:  Cuando el costo computacional importa (deployment, móvil, edge)
- FIC + AIC: Comparar ambos para decisión informada
""")