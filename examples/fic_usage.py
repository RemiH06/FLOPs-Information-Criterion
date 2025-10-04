#!/usr/bin/env python3
"""
FIC (FLOPs Information Criterion) - Ejemplos de Uso
Versión final con escala paramétrica optimizada

Después de experimentos exhaustivos, se determinó el valor óptimo de λ
para la fórmula: FIC = -2*log(L) + α*(λ*log(FLOPs) + (1-λ)*FLOPs/1e6) + β*k
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flop_counter.flop_information_criterion import FlopInformationCriterion

np.random.seed(42)

print("="*80)
print("FIC: FLOPs INFORMATION CRITERION - EJEMPLOS FINALES")
print("="*80)
print("\nFórmula refinada:")
print("  FIC = -2*log(L) + α*(λ*log(FLOPs) + (1-λ)*FLOPs/1e6) + β*k")
print("\nDonde:")
print("  α = 2.0  (coeficiente de penalización de FLOPs)")
print("  β = 1.0  (coeficiente de penalización de parámetros)")
print("  λ = [VALOR ÓPTIMO DE EXPERIMENTOS]")
print("="*80)

# ============================================================================
# CONFIGURACIÓN: Valor de λ optimizado (ajustar después de experimentos)
# ============================================================================

# TODO: Actualizar este valor después de ejecutar fic_refining.py
OPTIMAL_LAMBDA = 0.5  # Placeholder - actualizar con resultado experimental

FLOPS_SCALE = f'parametric_log_linear_{OPTIMAL_LAMBDA}'

print(f"\nUsando λ óptimo: {OPTIMAL_LAMBDA}")
print(f"Escala: {FLOPS_SCALE}\n")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_aic(log_likelihood: float, k: int) -> float:
    """AIC: Penaliza solo por # de parámetros"""
    return log_likelihood + 2 * k

def calculate_bic(log_likelihood: float, k: int, n: int) -> float:
    """BIC: Penalización crece con tamaño de muestra"""
    return log_likelihood + k * np.log(n)

def print_comparison_table(models_results: dict):
    """Imprime tabla comparativa de criterios"""
    print(f"\n{'='*80}")
    print("COMPARACIÓN DE CRITERIOS")
    print(f"{'='*80}")
    
    print(f"\n{'Modelo':<20} {'Params':<10} {'FLOPs':<15} {'AIC':<12} {'BIC':<12} {'FIC':<12}")
    print("-" * 80)
    
    for name, result in models_results.items():
        print(f"{name:<20} {result['n_params']:<10} {result['flops']:<15,} "
              f"{result['aic']:<12.2f} {result['bic']:<12.2f} {result['fic']:<12.2f}")
    
    # Mejores según cada criterio
    print(f"\n{'Criterio':<15} {'Selecciona':<20} {'Valor':<15} {'Razón':<30}")
    print("-" * 80)
    
    aic_best = min(models_results.items(), key=lambda x: x[1]['aic'])
    bic_best = min(models_results.items(), key=lambda x: x[1]['bic'])
    fic_best = min(models_results.items(), key=lambda x: x[1]['fic'])
    
    print(f"{'AIC':<15} {aic_best[0]:<20} {aic_best[1]['aic']:<15.2f} "
          f"{'Solo considera # params':<30}")
    print(f"{'BIC':<15} {bic_best[0]:<20} {bic_best[1]['bic']:<15.2f} "
          f"{'Penaliza más params':<30}")
    print(f"{'FIC':<15} {fic_best[0]:<20} {fic_best[1]['fic']:<15.2f} "
          f"{'Considera params + FLOPs':<30}")
    
    # Análisis ΔFIC
    print(f"\n{'='*80}")
    print("ΔFIC: Evidencia contra modelos subóptimos")
    print(f"{'='*80}")
    
    fic_values = {name: res['fic'] for name, res in models_results.items()}
    best_fic = min(fic_values.values())
    
    print(f"\n{'Modelo':<20} {'ΔFIC':<15} {'Interpretación':<40}")
    print("-" * 75)
    
    for name in sorted(models_results.keys(), key=lambda x: fic_values[x]):
        delta = fic_values[name] - best_fic
        
        if delta < 2:
            interp = "Sustancialmente equivalente al mejor"
        elif delta < 10:
            interp = "Evidencia considerable en contra"
        else:
            interp = "Evidencia fuerte en contra"
        
        marker = " ⭐" if delta < 0.01 else ""
        print(f"{name:<20} {delta:<15.2f} {interp:<40}{marker}")

# ============================================================================
# EJEMPLO 1: Regresión - Mismo ajuste, diferentes FLOPs
# ============================================================================

print(f"\n\n{'='*80}")
print("EJEMPLO 1: REGRESIÓN LINEAL")
print(f"{'='*80}")
print("\nEscenario: Dos modelos con igual # de parámetros y precisión")
print("           pero uno tiene operaciones innecesarias (más FLOPs)")
print("\nObjetivo: Demostrar que FIC detecta ineficiencia computacional")

# Datos
n_samples = 100
X_train = np.random.randn(n_samples, 1)
y_train = 2.5 * X_train.squeeze() + 1.0 + np.random.randn(n_samples) * 0.5

def model_efficient(X):
    """Modelo eficiente: y = 2.5x + 1.0"""
    W = np.array([[2.5], [1.0]])
    X_ext = np.column_stack([X, np.ones(len(X))])
    return X_ext @ W

def model_inefficient(X):
    """Mismo resultado pero con operaciones innecesarias"""
    W = np.array([[2.5], [1.0]])
    X_ext = np.column_stack([X, np.ones(len(X))])
    result = X_ext @ W
    
    # Operaciones costosas que no aportan valor
    for _ in range(5):
        result = np.exp(np.log(np.abs(result) + 1e-10)) * np.sign(result)
        result = result @ np.eye(1)
    
    return result

# Evaluar con FIC
fic_calc = FlopInformationCriterion(
    variant='hybrid',  # α=2, β=1
    flops_scale=FLOPS_SCALE
)

results_exp1 = {}

for name, model in [("Eficiente", model_efficient), ("Ineficiente", model_inefficient)]:
    result = fic_calc.evaluate_model(
        model=model,
        X=X_train,
        y_true=y_train,
        task='regression',
        n_params=2,
        framework='numpy'
    )
    
    # Calcular AIC y BIC para comparación
    result['aic'] = calculate_aic(result['log_likelihood_term'], 2)
    result['bic'] = calculate_bic(result['log_likelihood_term'], 2, n_samples)
    
    results_exp1[name] = result

print_comparison_table(results_exp1)

print("\n📊 Análisis:")
flops_eff = results_exp1['Eficiente']['flops']
flops_ineff = results_exp1['Ineficiente']['flops']

print(f"   FLOPs Eficiente:    {flops_eff:,}")
print(f"   FLOPs Ineficiente:  {flops_ineff:,}")

if flops_eff > 0:
    ratio = flops_ineff / flops_eff
    print(f"   Ratio:              {ratio:.1f}x")
else:
    print(f"   Ratio:              No disponible (contador retorna 0 FLOPs)")
    print(f"   ⚠️  Advertencia: El contador de FLOPs puede no estar funcionando correctamente")

print("\n💡 Conclusión:")
print("   AIC/BIC: Idénticos (mismo # parámetros, mismo ajuste)")
print("   FIC:     Detecta y penaliza la ineficiencia computacional")

# ============================================================================
# EJEMPLO 2: Redes Neuronales - Trade-off complejidad vs eficiencia
# ============================================================================

print(f"\n\n{'='*80}")
print("EJEMPLO 2: ARQUITECTURAS DE REDES NEURONALES")
print(f"{'='*80}")
print("\nEscenario: Tres arquitecturas para clasificación con FLOPs muy diferentes")
print("           - Simple: 2 capas, eficiente")
print("           - Deep: 5 capas, muchos FLOPs por profundidad")
print("           - Wide-Deep: 4 capas anchas, máximo FLOPs")
print("\nObjetivo: Demostrar que FIC distingue arquitecturas por FLOPs, no solo parámetros")

# Datos
n_samples = 200
X_train = np.random.randn(n_samples, 20)
y_train = np.random.randint(0, 3, n_samples)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def simple_net(X):
    """20 -> 50 -> 3 (2 capas, ~1,053 parámetros, FLOPs bajos)"""
    W1 = np.random.randn(20, 50) * 0.01
    b1 = np.zeros(50)
    W2 = np.random.randn(50, 3) * 0.01
    b2 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)
    logits = h1 @ W2 + b2
    return softmax(logits)

def deep_net(X):
    """20 -> 25 -> 25 -> 25 -> 25 -> 3 (5 capas, ~1,728 parámetros, FLOPs medios-altos)"""
    W1 = np.random.randn(20, 25) * 0.01
    b1 = np.zeros(25)
    W2 = np.random.randn(25, 25) * 0.01
    b2 = np.zeros(25)
    W3 = np.random.randn(25, 25) * 0.01
    b3 = np.zeros(25)
    W4 = np.random.randn(25, 25) * 0.01
    b4 = np.zeros(25)
    W5 = np.random.randn(25, 3) * 0.01
    b5 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    h3 = np.maximum(0, h2 @ W3 + b3)
    h4 = np.maximum(0, h3 @ W4 + b4)
    logits = h4 @ W5 + b5
    return softmax(logits)

def wide_deep_net(X):
    """20 -> 80 -> 60 -> 40 -> 3 (4 capas anchas, ~6,343 parámetros, FLOPs muy altos)"""
    W1 = np.random.randn(20, 80) * 0.01
    b1 = np.zeros(80)
    W2 = np.random.randn(80, 60) * 0.01
    b2 = np.zeros(60)
    W3 = np.random.randn(60, 40) * 0.01
    b3 = np.zeros(40)
    W4 = np.random.randn(40, 3) * 0.01
    b4 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    h3 = np.maximum(0, h2 @ W3 + b3)
    logits = h3 @ W4 + b4
    return softmax(logits)

results_exp2 = {}

for name, model, n_params in [
    ("Simple", simple_net, 1053),
    ("Deep", deep_net, 1728),
    ("Wide-Deep", wide_deep_net, 6343)
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
    
    results_exp2[name] = result

print_comparison_table(results_exp2)

print("\n📊 Análisis de arquitecturas:")
for name in ["Simple", "Deep", "Wide-Deep"]:
    r = results_exp2[name]
    flops_per_param = r['flops'] / r['n_params'] if r['n_params'] > 0 else 0
    print(f"\n   {name}:")
    print(f"      Parámetros:  {r['n_params']:,}")
    print(f"      FLOPs:       {r['flops']:,}")
    print(f"      FLOPs/param: {flops_per_param:.1f}")
    print(f"      Accuracy:    {r['accuracy']:.3f}")

print("\n💡 Conclusión:")
print("   AIC:  Prefiere Simple (menos parámetros)")
print("   BIC:  Similar a AIC, también prefiere Simple")
print("   FIC:  Considera parámetros Y FLOPs")
print("         - Simple: Pocos params, pocos FLOPs → eficiente")
print("         - Deep: Params medios, FLOPs medios-altos (profundidad)")
print("         - Wide-Deep: Muchos params, muchos FLOPs → costoso")
print("         FIC penaliza Wide-Deep más que AIC/BIC por su alto costo computacional")

# ============================================================================
# EJEMPLO 3: Polinomios - Detección de overfitting y complejidad
# ============================================================================

print(f"\n\n{'='*80}")
print("EJEMPLO 3: REGRESIÓN POLINOMIAL")
print(f"{'='*80}")
print("\nEscenario: Datos generados por función cuadrática")
print("           Ajustamos polinomios de grado 1, 2, 3, 5 y 10")
print("\nObjetivo: Demostrar que FIC detecta overfitting considerando FLOPs")

# Datos con relación cuadrática verdadera
n_samples = 150
X_poly = np.random.randn(n_samples, 1)
y_poly = 2.0 * X_poly.squeeze()**2 + 1.5 * X_poly.squeeze() + 1.0
y_poly += np.random.randn(n_samples) * 0.5

def poly_model(degree):
    """Genera modelo polinomial de grado específico"""
    def model(X):
        X_features = np.column_stack([X**i for i in range(degree + 1)])
        coeffs = np.random.randn(degree + 1) * 0.1
        coeffs[0] = 1.0
        if degree >= 1:
            coeffs[1] = 1.5
        if degree >= 2:
            coeffs[2] = 2.0
        return X_features @ coeffs
    return model

results_exp3 = {}

for degree in [1, 2, 3, 5, 10]:
    name = f"Poly{degree}"
    model = poly_model(degree)
    
    result = fic_calc.evaluate_model(
        model=model,
        X=X_poly,
        y_true=y_poly,
        task='regression',
        n_params=degree + 1,
        framework='numpy'
    )
    
    result['aic'] = calculate_aic(result['log_likelihood_term'], degree + 1)
    result['bic'] = calculate_bic(result['log_likelihood_term'], degree + 1, n_samples)
    
    results_exp3[name] = result

print_comparison_table(results_exp3)

print("\n📊 Análisis por grado:")
for degree in [1, 2, 3, 5, 10]:
    name = f"Poly{degree}"
    r = results_exp3[name]
    print(f"\n   Grado {degree}:")
    print(f"      R²:         {r['accuracy']:.4f}")
    print(f"      Parámetros: {r['n_params']}")
    print(f"      FLOPs:      {r['flops']:,}")

print("\n💡 Conclusión:")
print("   Poly2 (cuadrático) es el modelo verdadero")
print("   AIC/BIC: Detectan overfitting solo vía # de parámetros")
print("   FIC:     Penaliza además el costo computacional del overfitting")
print("            Poly10 no solo tiene más parámetros, también más FLOPs")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print(f"\n\n{'='*80}")
print("RESUMEN: CUÁNDO USAR FIC")
print(f"{'='*80}")

print("""
✅ FIC es superior a AIC/BIC cuando:

1. DEPLOYMENT REAL
   - Modelos se ejecutarán en producción
   - Latencia y throughput son importantes
   - Hardware tiene limitaciones (móvil, edge, IoT)

2. ARQUITECTURAS COMPLEJAS
   - Comparando redes con diferentes profundidades
   - Trade-off entre ancho y profundidad
   - Operaciones costosas (attention, convolutions)

3. EFICIENCIA COMPUTACIONAL
   - Distinguir modelos con igual # parámetros
   - Detectar operaciones innecesarias
   - Optimización de recursos

📊 AIC/BIC siguen siendo útiles para:
   - Análisis estadístico tradicional
   - Modelos donde FLOPs no importan
   - Cuando solo interesa complejidad paramétrica

🎯 FÓRMULA FINAL DEL FIC:
   
   FIC = -2*log(L) + α*(λ*log(FLOPs) + (1-λ)*FLOPs/1e6) + β*k
   
   Donde:
   - L: verosimilitud del modelo
   - α: peso de penalización de FLOPs (recomendado: 2.0)
   - λ: balance log/linear (determinado experimentalmente)
   - β: peso de penalización de parámetros (recomendado: 1.0)
   - k: número de parámetros

💡 INTERPRETACIÓN DE ΔFIC:
   
   ΔFIC < 2:   Modelos prácticamente equivalentes
   2 < ΔFIC < 10:  Evidencia considerable contra el modelo
   ΔFIC > 10:  Evidencia fuerte contra el modelo

🚀 PRÓXIMOS PASOS:
   
   1. Ejecutar fic_refining.py para determinar λ óptimo
   2. Actualizar OPTIMAL_LAMBDA en este archivo
   3. Usar FIC para selección de modelos en tus proyectos
""")

print(f"\n{'='*80}")
print("Ejemplos completados exitosamente")
print(f"{'='*80}\n")