#!/usr/bin/env python3
"""
FIC: Experimentación con escalas paramétricas log-linear
Objetivo: Encontrar el mejor balance λ entre log y linear
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from flop_counter.flop_information_criterion import FlopInformationCriterion

np.random.seed(42)

print("="*100)
print("FIC: EXPERIMENTACIÓN CON ESCALAS PARAMÉTRICAS")
print("="*100)

# ============================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================================

# Valores de λ para explorar (balance log vs linear)
LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# Escalas a comparar:
# - log_plus_linear (original, λ fijo)
# - parametric con diferentes λ
ALL_SCALES = ['log_plus_linear'] + [f'parametric_log_linear_{lam}' for lam in LAMBDA_VALUES]

# Valores de α (penalización FLOPs) - reducidos para acelerar
ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]

# Valores de β (penalización parámetros) - reducidos
BETA_VALUES = [0.0, 0.5, 1.0, 2.0]

print(f"\nConfiguración:")
print(f"  Escalas paramétricas: {len(LAMBDA_VALUES)} valores de λ")
print(f"  Valores α: {ALPHA_VALUES}")
print(f"  Valores β: {BETA_VALUES}")
print(f"  Total configuraciones por experimento: {len(ALL_SCALES) * len(ALPHA_VALUES) * len(BETA_VALUES)}")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_aic(log_likelihood: float, k: int) -> float:
    return log_likelihood + 2 * k

def calculate_bic(log_likelihood: float, k: int, n: int) -> float:
    return log_likelihood + k * np.log(n)

def create_results_dataframe(results_dict):
    """Convierte resultados a DataFrame."""
    data = []
    for key, result in results_dict.items():
        scale, alpha, beta, model = key
        
        # Extraer λ del nombre de la escala
        if scale.startswith('parametric_log_linear_'):
            lambda_val = float(scale.split('_')[-1])
            scale_type = 'parametric'
        elif scale == 'log_plus_linear':
            lambda_val = None  # No aplica
            scale_type = 'original'
        else:
            lambda_val = None
            scale_type = 'other'
        
        data.append({
            'Scale': scale,
            'ScaleType': scale_type,
            'Lambda': lambda_val,
            'Alpha': alpha,
            'Beta': beta,
            'Model': model,
            'FIC': result['fic'],
            'LogLik': result['log_likelihood_term'],
            'FLOPs_Penalty': result['flops_penalty'],
            'Params_Penalty': result['params_penalty'],
            'FLOPs': result['flops'],
            'Params': result['n_params'],
            'Accuracy': result.get('accuracy', 0)
        })
    return pd.DataFrame(data)

def analyze_lambda_sensitivity(df, experiment_name):
    """Analiza cómo λ afecta la sensibilidad del FIC."""
    print(f"\n{'='*100}")
    print(f"ANÁLISIS DE SENSIBILIDAD: {experiment_name}")
    print(f"{'='*100}")
    
    # Solo analizar escalas paramétricas
    param_df = df[df['ScaleType'] == 'parametric'].copy()
    
    if len(param_df) == 0:
        print("No hay datos de escalas paramétricas")
        return
    
    # Determinar modelo de referencia
    all_models = df['Model'].unique()
    reference_model = 'Simple' if 'Simple' in all_models else sorted(all_models)[0]
    
    print(f"\n📊 SENSIBILIDAD POR VALOR DE λ (referencia: {reference_model}):")
    print(f"{'λ':<10} {'Avg ΔFIC':<15} {'Std ΔFIC':<15} {'Max ΔFIC':<15} {'Best α':<10} {'Best β':<10}")
    print("-" * 85)
    
    lambda_stats = []
    
    for lambda_val in sorted(param_df['Lambda'].unique()):
        lambda_df = param_df[param_df['Lambda'] == lambda_val]
        
        # Calcular sensibilidad para esta λ
        deltas = []
        for alpha in ALPHA_VALUES:
            for beta in BETA_VALUES:
                config_df = lambda_df[(lambda_df['Alpha'] == alpha) & (lambda_df['Beta'] == beta)]
                
                if len(config_df) >= 2:
                    ref_config = config_df[config_df['Model'] == reference_model]
                    if len(ref_config) > 0:
                        ref_fic = ref_config['FIC'].values[0]
                        other_fics = config_df[config_df['Model'] != reference_model]['FIC'].values
                        for other_fic in other_fics:
                            deltas.append(abs(other_fic - ref_fic))
        
        if deltas:
            avg_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            max_delta = np.max(deltas)
            
            # Encontrar mejor configuración para esta λ
            best_config = lambda_df.groupby(['Alpha', 'Beta']).apply(
                lambda x: abs(x[x['Model'] != reference_model]['FIC'].mean() - 
                             x[x['Model'] == reference_model]['FIC'].mean())
            ).idxmax()
            
            lambda_stats.append({
                'lambda': lambda_val,
                'avg_delta': avg_delta,
                'std_delta': std_delta,
                'max_delta': max_delta,
                'best_alpha': best_config[0],
                'best_beta': best_config[1]
            })
            
            print(f"{lambda_val:<10.1f} {avg_delta:<15.2f} {std_delta:<15.2f} {max_delta:<15.2f} "
                  f"α={best_config[0]:<8.1f} β={best_config[1]:<8.1f}")
    
    # Encontrar mejor λ
    if lambda_stats:
        best_lambda_stat = max(lambda_stats, key=lambda x: x['avg_delta'])
        worst_lambda_stat = min(lambda_stats, key=lambda x: x['avg_delta'])
        
        print(f"\n💡 RESULTADOS:")
        print(f"   ⭐ Mejor λ: {best_lambda_stat['lambda']:.1f} "
              f"(ΔFIC promedio = {best_lambda_stat['avg_delta']:.2f})")
        print(f"      Configuración óptima: α={best_lambda_stat['best_alpha']:.1f}, "
              f"β={best_lambda_stat['best_beta']:.1f}")
        print(f"   ⚠️  Peor λ: {worst_lambda_stat['lambda']:.1f} "
              f"(ΔFIC promedio = {worst_lambda_stat['avg_delta']:.2f})")
        
        # Comparar con log_plus_linear original
        original_df = df[df['Scale'] == 'log_plus_linear']
        if len(original_df) > 0:
            orig_deltas = []
            for alpha in ALPHA_VALUES:
                for beta in BETA_VALUES:
                    config_df = original_df[(original_df['Alpha'] == alpha) & (original_df['Beta'] == beta)]
                    if len(config_df) >= 2:
                        ref_config = config_df[config_df['Model'] == reference_model]
                        if len(ref_config) > 0:
                            ref_fic = ref_config['FIC'].values[0]
                            other_fics = config_df[config_df['Model'] != reference_model]['FIC'].values
                            for other_fic in other_fics:
                                orig_deltas.append(abs(other_fic - ref_fic))
            
            if orig_deltas:
                orig_avg = np.mean(orig_deltas)
                improvement = ((best_lambda_stat['avg_delta'] - orig_avg) / orig_avg) * 100
                
                print(f"\n   📈 Comparación con log_plus_linear original:")
                print(f"      Original ΔFIC: {orig_avg:.2f}")
                print(f"      Mejor paramétrico: {best_lambda_stat['avg_delta']:.2f}")
                print(f"      Mejora: {improvement:+.1f}%")
    
    return lambda_stats

def print_experiment_header(title, models_info):
    """Imprime header de experimento."""
    print(f"\n\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print("\n📋 Modelos a comparar:")
    for model_name, info in models_info.items():
        print(f"   - {model_name}: {info['params']} parámetros, {info['expected_flops']} FLOPs esperados")

# ============================================================================
# EJEMPLO 1: REGRESIÓN LINEAL - MODELO SIMPLE VS COMPLEJO
# ============================================================================

print_experiment_header(
    "EXPERIMENTO 1: REGRESIÓN LINEAL",
    {
        'Simple': {'params': 2, 'expected_flops': '~500'},
        'Complejo': {'params': 2, 'expected_flops': '~5000 (10x más)'}
    }
)

# Datos
n_samples = 100
X_train = np.random.randn(n_samples, 1)
true_slope = 2.5
true_intercept = 1.0
noise = np.random.randn(n_samples) * 0.5
y_train = true_slope * X_train.squeeze() + true_intercept + noise

def linear_model_simple(X):
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    return X_extended @ W

def linear_model_complex(X):
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    temp = X_extended @ W
    
    # Operaciones costosas e innecesarias
    for _ in range(5):
        temp = np.exp(np.log(np.abs(temp) + 1e-10)) * np.sign(temp)
        temp = temp @ np.eye(1)
        temp = np.sin(np.arcsin(np.clip(temp / (np.max(np.abs(temp)) + 1e-10), -0.99, 0.99))) * (np.max(np.abs(temp)) + 1e-10)
    
    return temp

print("\n🔬 Ejecutando experimentos...")
print(f"   Total configuraciones: {len(ALL_SCALES) * len(ALPHA_VALUES) * len(BETA_VALUES)}")

results_exp1 = {}

for scale in ALL_SCALES:
    scale_label = scale if scale == 'log_plus_linear' else f"λ={scale.split('_')[-1]}"
    print(f"   Procesando: {scale_label}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            # Para escalas paramétricas, usar variant='custom'
            if scale.startswith('parametric_log_linear_'):
                fic_calc = FlopInformationCriterion(
                    variant='custom', 
                    alpha=alpha, 
                    beta=beta, 
                    flops_scale=scale
                )
            else:
                fic_calc = FlopInformationCriterion(
                    variant='custom',
                    alpha=alpha,
                    beta=beta,
                    flops_scale=scale
                )
            
            # Evaluar modelos
            try:
                result_simple = fic_calc.evaluate_model(
                    model=linear_model_simple,
                    X=X_train,
                    y_true=y_train,
                    task='regression',
                    n_params=2,
                    framework='numpy',
                    alpha_override=alpha,
                    beta_override=beta
                )
                
                result_complex = fic_calc.evaluate_model(
                    model=linear_model_complex,
                    X=X_train,
                    y_true=y_train,
                    task='regression',
                    n_params=2,
                    framework='numpy',
                    alpha_override=alpha,
                    beta_override=beta
                )
                
                results_exp1[(scale, alpha, beta, 'Simple')] = result_simple
                results_exp1[(scale, alpha, beta, 'Complejo')] = result_complex
                
            except Exception as e:
                print(f"      ⚠️  Error en {scale}, α={alpha}, β={beta}: {e}")
                continue

# Analizar resultados
df_exp1 = create_results_dataframe(results_exp1)
lambda_stats_exp1 = analyze_lambda_sensitivity(df_exp1, "Experimento 1: Regresión Lineal")

# ============================================================================
# EJEMPLO 2: REDES NEURONALES
# ============================================================================

print_experiment_header(
    "EXPERIMENTO 2: REDES NEURONALES",
    {
        'Wide-Shallow': {'params': 2303, 'expected_flops': 'bajo'},
        'Deep-Narrow': {'params': 2523, 'expected_flops': 'alto'},
        'Balanced': {'params': 2353, 'expected_flops': 'medio'}
    }
)

# Datos
n_samples = 200
n_features = 20
n_classes = 3
X_train = np.random.randn(n_samples, n_features)
y_train = np.random.randint(0, n_classes, n_samples)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def wide_shallow_net(X):
    W1 = np.random.randn(20, 100) * 0.01
    b1 = np.zeros(100)
    W2 = np.random.randn(100, 3) * 0.01
    b2 = np.zeros(3)
    h1 = np.maximum(0, X @ W1 + b1)
    logits = h1 @ W2 + b2
    return softmax(logits)

def deep_narrow_net(X):
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

n_params_wide = (20*100 + 100) + (100*3 + 3)
n_params_deep = (20*30 + 30) + (30*30 + 30) + (30*30 + 30) + (30*3 + 3)
n_params_balanced = (20*50 + 50) + (50*25 + 25) + (25*3 + 3)

print("\n🔬 Ejecutando experimentos...")

results_exp2 = {}

for scale in ALL_SCALES:
    scale_label = scale if scale == 'log_plus_linear' else f"λ={scale.split('_')[-1]}"
    print(f"   Procesando: {scale_label}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            fic_calc = FlopInformationCriterion(
                variant='custom',
                alpha=alpha,
                beta=beta,
                flops_scale=scale
            )
            
            for name, model, n_params in [
                ("Wide-Shallow", wide_shallow_net, n_params_wide),
                ("Deep-Narrow", deep_narrow_net, n_params_deep),
                ("Balanced", balanced_net, n_params_balanced)
            ]:
                try:
                    result = fic_calc.evaluate_model(
                        model=model,
                        X=X_train,
                        y_true=y_train,
                        task='classification',
                        n_params=n_params,
                        framework='numpy',
                        alpha_override=alpha,
                        beta_override=beta
                    )
                    
                    results_exp2[(scale, alpha, beta, name)] = result
                    
                except Exception as e:
                    print(f"      ⚠️  Error en {name}: {e}")
                    continue

df_exp2 = create_results_dataframe(results_exp2)
lambda_stats_exp2 = analyze_lambda_sensitivity(df_exp2, "Experimento 2: Redes Neuronales")

# ============================================================================
# EJEMPLO 3: POLINOMIOS
# ============================================================================

print_experiment_header(
    "EXPERIMENTO 3: REGRESIÓN POLINOMIAL",
    {
        'Linear': {'params': 2, 'expected_flops': 'muy bajo'},
        'Quadratic': {'params': 3, 'expected_flops': 'bajo (óptimo)'},
        'Cubic': {'params': 4, 'expected_flops': 'medio'},
        'Poly5': {'params': 6, 'expected_flops': 'alto'},
        'Poly10': {'params': 11, 'expected_flops': 'muy alto'}
    }
)

# Datos con relación cuadrática
n_samples = 150
X_poly = np.random.randn(n_samples, 1)
y_poly = 2.0 * X_poly.squeeze()**2 + 1.5 * X_poly.squeeze() + 1.0 + np.random.randn(n_samples) * 0.5

def poly_model(X, degree):
    """Modelo polinomial de grado especificado."""
    def model(X_input):
        X_features = np.column_stack([X_input**i for i in range(degree + 1)])
        coeffs = np.random.randn(degree + 1) * 0.1
        coeffs[0] = 1.0
        if degree >= 1:
            coeffs[1] = 1.5
        if degree >= 2:
            coeffs[2] = 2.0
        return X_features @ coeffs
    return model

print("\n🔬 Ejecutando experimentos...")

results_exp3 = {}

for scale in ALL_SCALES:
    scale_label = scale if scale == 'log_plus_linear' else f"λ={scale.split('_')[-1]}"
    print(f"   Procesando: {scale_label}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            fic_calc = FlopInformationCriterion(
                variant='custom',
                alpha=alpha,
                beta=beta,
                flops_scale=scale
            )
            
            for degree, name in [(1, 'Linear'), (2, 'Quadratic'), (3, 'Cubic'), 
                                  (5, 'Poly5'), (10, 'Poly10')]:
                try:
                    model = poly_model(X_poly, degree)
                    
                    result = fic_calc.evaluate_model(
                        model=model,
                        X=X_poly,
                        y_true=y_poly,
                        task='regression',
                        n_params=degree + 1,
                        framework='numpy',
                        alpha_override=alpha,
                        beta_override=beta
                    )
                    
                    results_exp3[(scale, alpha, beta, name)] = result
                    
                except Exception as e:
                    print(f"      ⚠️  Error en {name}: {e}")
                    continue

df_exp3 = create_results_dataframe(results_exp3)
lambda_stats_exp3 = analyze_lambda_sensitivity(df_exp3, "Experimento 3: Regresión Polinomial")

# ============================================================================
# ANÁLISIS FINAL
# ============================================================================

print(f"\n\n{'='*100}")
print("RESUMEN FINAL: MEJOR VALOR DE λ")
print(f"{'='*100}")

if lambda_stats_exp1 and lambda_stats_exp2 and lambda_stats_exp3:
    # Calcular λ promedio óptimo
    best_lambdas = [
        max(lambda_stats_exp1, key=lambda x: x['avg_delta'])['lambda'],
        max(lambda_stats_exp2, key=lambda x: x['avg_delta'])['lambda'],
        max(lambda_stats_exp3, key=lambda x: x['avg_delta'])['lambda']
    ]
    
    print(f"\n📊 Mejor λ por experimento:")
    print(f"   Exp 1 (Regresión): λ = {best_lambdas[0]:.1f}")
    print(f"   Exp 2 (Redes): λ = {best_lambdas[1]:.1f}")
    print(f"   Exp 3 (Polinomios): λ = {best_lambdas[2]:.1f}")
    
    avg_lambda = np.mean(best_lambdas)
    print(f"\n⭐ λ ÓPTIMO PROMEDIO: {avg_lambda:.2f}")
    
    print(f"\n💡 FÓRMULA RECOMENDADA:")
    print(f"   FIC = -2*log(L) + α*(λ*log(FLOPs) + (1-λ)*FLOPs/1e6) + β*k")
    print(f"   donde λ = {avg_lambda:.2f}")
    print(f"\n   Interpretación:")
    if avg_lambda > 0.7:
        print(f"   - λ alto ({avg_lambda:.2f}): Domina término logarítmico")
        print(f"   - Sensible a diferencias relativas en FLOPs")
    elif avg_lambda < 0.3:
        print(f"   - λ bajo ({avg_lambda:.2f}): Domina término lineal")
        print(f"   - Sensible a diferencias absolutas en FLOPs")
    else:
        print(f"   - λ balanceado ({avg_lambda:.2f}): Equilibrio log-linear")
        print(f"   - Sensible a ambos tipos de diferencias")

print(f"\n{'='*100}")
print("EXPERIMENTOS COMPLETADOS")
print(f"{'='*100}")