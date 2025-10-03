#!/usr/bin/env python3
"""
FIC: Experimentación exhaustiva con todas las escalas y coeficientes
Objetivo: Encontrar la mejor combinación de escala, α y β para el FIC
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
print("FIC: EXPERIMENTACIÓN EXHAUSTIVA - TODAS LAS ESCALAS Y COEFICIENTES")
print("="*100)

# ============================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================================

# Escalas seleccionadas para experimentación
ALL_SCALES = [
    'log_plus_linear',      # log(FLOPs) + FLOPs/1e6 - Híbrido
    'sqrt_mega',            # sqrt(FLOPs / 1e6) - Suave
    'log_normalized',       # log(FLOPs / 1e3) - Normalizado
    'cube_root_kilo',       # (FLOPs^(1/3)) / 1e3 - Nueva sugerencia 1
    'log_flops_per_param'   # log(1 + FLOPs/params) - Nueva sugerencia 2
]

# Valores de α (coeficiente FLOPs) para experimentar
ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]

# Valores de β (coeficiente parámetros) para experimentar
BETA_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_aic(log_likelihood: float, k: int) -> float:
    return log_likelihood + 2 * k

def calculate_bic(log_likelihood: float, k: int, n: int) -> float:
    return log_likelihood + k * np.log(n)

def create_results_dataframe(results_dict):
    """Convierte resultados a DataFrame para análisis."""
    data = []
    for key, result in results_dict.items():
        scale, alpha, beta, model = key
        data.append({
            'Scale': scale,
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

def analyze_experiment(df, experiment_name):
    """Analiza resultados de un experimento."""
    print(f"\n{'='*100}")
    print(f"ANÁLISIS: {experiment_name}")
    print(f"{'='*100}")
    
    # Determinar el modelo de referencia (el primero alfabéticamente o 'Simple' si existe)
    all_models = df['Model'].unique()
    reference_model = 'Simple' if 'Simple' in all_models else sorted(all_models)[0]
    
    # Para cada escala, encontrar la mejor combinación α, β
    print(f"\n📊 MEJORES CONFIGURACIONES POR ESCALA (referencia: {reference_model}):")
    print(f"{'Scale':<20} {'Best α':<10} {'Best β':<10} {'Model':<15} {'FIC':<12} {'ΔFIC vs Ref':<15}")
    print("-" * 100)
    
    for scale in ALL_SCALES:
        scale_df = df[df['Scale'] == scale]
        
        # Encontrar el modelo de referencia con mejor FIC para esta escala
        ref_df = scale_df[scale_df['Model'] == reference_model]
        if len(ref_df) == 0:
            continue
        
        best_ref = ref_df.loc[ref_df['FIC'].idxmin()]
        
        # Para cada modelo, ver qué tan bien lo distingue
        for model in scale_df['Model'].unique():
            if model == reference_model:
                continue
            
            model_df = scale_df[scale_df['Model'] == model]
            best_model = model_df.loc[model_df['FIC'].idxmin()]
            
            delta_fic = best_model['FIC'] - best_ref['FIC']
            
            print(f"{scale:<20} α={best_model['Alpha']:<8.1f} β={best_model['Beta']:<8.1f} "
                  f"{model:<15} {best_model['FIC']:<12.2f} {delta_fic:>+14.2f}")
    
    # Analizar sensibilidad de cada escala
    print(f"\n📈 SENSIBILIDAD DE ESCALAS (diferencia promedio entre modelos):")
    print(f"{'Scale':<20} {'Avg ΔFIC':<15} {'Std ΔFIC':<15} {'Max ΔFIC':<15}")
    print("-" * 70)
    
    for scale in ALL_SCALES:
        scale_df = df[df['Scale'] == scale]
        
        # Calcular diferencias entre modelo de referencia y otros modelos
        ref_fics = scale_df[scale_df['Model'] == reference_model]['FIC'].values
        other_fics = scale_df[scale_df['Model'] != reference_model]['FIC'].values
        
        if len(ref_fics) == 0 or len(other_fics) == 0:
            continue
        
        # Para cada α,β calcular la diferencia promedio
        deltas = []
        for alpha in ALPHA_VALUES:
            for beta in BETA_VALUES:
                config_df = scale_df[(scale_df['Alpha'] == alpha) & (scale_df['Beta'] == beta)]
                if len(config_df) >= 2:
                    ref_config = config_df[config_df['Model'] == reference_model]
                    if len(ref_config) > 0:
                        ref_fic = ref_config['FIC'].values[0]
                        other_fic = config_df[config_df['Model'] != reference_model]['FIC'].mean()
                        deltas.append(abs(other_fic - ref_fic))
        
        if deltas:
            print(f"{scale:<20} {np.mean(deltas):<15.2f} {np.std(deltas):<15.2f} {np.max(deltas):<15.2f}")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    
    # Escala con mayor sensibilidad
    sensitivities = {}
    for scale in ALL_SCALES:
        scale_df = df[df['Scale'] == scale]
        deltas = []
        for alpha in ALPHA_VALUES:
            for beta in BETA_VALUES:
                config_df = scale_df[(scale_df['Alpha'] == alpha) & (scale_df['Beta'] == beta)]
                if len(config_df) >= 2:
                    ref_config = config_df[config_df['Model'] == reference_model]
                    if len(ref_config) > 0:
                        ref_fic = ref_config['FIC'].values[0]
                        other_fic = config_df[config_df['Model'] != reference_model]['FIC'].mean()
                        deltas.append(abs(other_fic - ref_fic))
        
        if deltas:
            sensitivities[scale] = np.mean(deltas)
    
    if sensitivities:
        best_scale = max(sensitivities, key=sensitivities.get)
        print(f"   - Escala más sensible: {best_scale} (ΔFIC promedio = {sensitivities[best_scale]:.2f})")
        
        worst_scale = min(sensitivities, key=sensitivities.get)
        print(f"   - Escala menos sensible: {worst_scale} (ΔFIC promedio = {sensitivities[worst_scale]:.2f})")

def print_experiment_header(title, models_info):
    """Imprime header de experimento."""
    print(f"\n\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print("\n📋 Modelos a comparar:")
    for model_name, info in models_info.items():
        print(f"   - {model_name}: {info['params']} parámetros, ~{info['expected_flops']} FLOPs esperados")

# ============================================================================
# EJEMPLO 1: REGRESIÓN LINEAL - MODELO SIMPLE VS COMPLEJO
# ============================================================================

print_experiment_header(
    "EXPERIMENTO 1: REGRESIÓN LINEAL",
    {
        'Simple': {'params': 2, 'expected_flops': '~500'},
        'Complejo': {'params': 2, 'expected_flops': '~5000 (operaciones innecesarias)'}
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
    
    # Muchas operaciones costosas e innecesarias
    for _ in range(5):
        temp = np.exp(np.log(np.abs(temp) + 1e-10)) * np.sign(temp)
        temp = temp @ np.eye(1)
        temp = np.sin(np.arcsin(np.clip(temp / (np.max(np.abs(temp)) + 1e-10), -0.99, 0.99))) * (np.max(np.abs(temp)) + 1e-10)
    
    return temp

print(f"\n🔬 Ejecutando experimentos...")
print(f"   Escalas seleccionadas: {len(ALL_SCALES)}")
print(f"   - log_plus_linear: log(FLOPs) + FLOPs/1e6")
print(f"   - sqrt_mega: sqrt(FLOPs/1e6)")
print(f"   - log_normalized: log(FLOPs/1e3)")
print(f"   - cube_root_kilo: (FLOPs^(1/3))/1e3 [NUEVA]")
print(f"   - log_flops_per_param: log(1 + FLOPs/params) [NUEVA]")
print(f"   Valores α: {ALPHA_VALUES}")
print(f"   Valores β: {BETA_VALUES}")
print(f"   Total configuraciones: {len(ALL_SCALES) * len(ALPHA_VALUES) * len(BETA_VALUES)}")

results_exp1 = {}

for scale in ALL_SCALES:
    print(f"\n   Procesando escala: {scale}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            fic_calc = FlopInformationCriterion(variant='custom', alpha=alpha, beta=beta, flops_scale=scale)
            
            # Evaluar modelo simple
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
            
            # Evaluar modelo complejo
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

# Analizar resultados
df_exp1 = create_results_dataframe(results_exp1)
analyze_experiment(df_exp1, "Experimento 1: Regresión Lineal")

# ============================================================================
# EJEMPLO 2: REDES NEURONALES - ARQUITECTURAS DIFERENTES
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
    print(f"\n   Procesando escala: {scale}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            fic_calc = FlopInformationCriterion(variant='custom', alpha=alpha, beta=beta, flops_scale=scale)
            
            for name, model, n_params in [
                ("Wide-Shallow", wide_shallow_net, n_params_wide),
                ("Deep-Narrow", deep_narrow_net, n_params_deep),
                ("Balanced", balanced_net, n_params_balanced)
            ]:
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

df_exp2 = create_results_dataframe(results_exp2)
analyze_experiment(df_exp2, "Experimento 2: Redes Neuronales")

# ============================================================================
# EJEMPLO 3: POLINOMIOS - COMPLEJIDAD CRECIENTE
# ============================================================================

print_experiment_header(
    "EXPERIMENTO 3: REGRESIÓN POLINOMIAL",
    {
        'Linear': {'params': 2, 'expected_flops': 'muy bajo'},
        'Quadratic': {'params': 3, 'expected_flops': 'bajo'},
        'Cubic': {'params': 4, 'expected_flops': 'medio'},
        'Poly5': {'params': 6, 'expected_flops': 'alto'},
        'Poly10': {'params': 11, 'expected_flops': 'muy alto'}
    }
)

# Datos con relación no lineal
n_samples = 150
X_poly = np.random.randn(n_samples, 1)
# Verdadera relación: cuadrática con ruido
y_poly = 2.0 * X_poly.squeeze()**2 + 1.5 * X_poly.squeeze() + 1.0 + np.random.randn(n_samples) * 0.5

def poly_model(X, degree):
    """Modelo polinomial de grado especificado."""
    def model(X_input):
        X_features = np.column_stack([X_input**i for i in range(degree + 1)])
        # Coeficientes aleatorios pero consistentes
        coeffs = np.random.randn(degree + 1) * 0.1
        coeffs[0] = 1.0  # intercept correcto
        if degree >= 1:
            coeffs[1] = 1.5  # término lineal correcto
        if degree >= 2:
            coeffs[2] = 2.0  # término cuadrático correcto
        return X_features @ coeffs
    return model

print("\n🔬 Ejecutando experimentos...")

results_exp3 = {}

for scale in ALL_SCALES:
    print(f"\n   Procesando escala: {scale}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            fic_calc = FlopInformationCriterion(variant='custom', alpha=alpha, beta=beta, flops_scale=scale)
            
            for degree, name in [(1, 'Linear'), (2, 'Quadratic'), (3, 'Cubic'), 
                                  (5, 'Poly5'), (10, 'Poly10')]:
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

df_exp3 = create_results_dataframe(results_exp3)
analyze_experiment(df_exp3, "Experimento 3: Regresión Polinomial")

# ============================================================================
# ANÁLISIS COMPARATIVO GLOBAL
# ============================================================================

print(f"\n\n{'='*100}")
print("ANÁLISIS COMPARATIVO GLOBAL: TODAS LAS ESCALAS EN TODOS LOS EXPERIMENTOS")
print(f"{'='*100}")

print("\n📊 RANKING DE ESCALAS POR SENSIBILIDAD (promedio en todos los experimentos):")

# Calcular sensibilidad promedio de cada escala
scale_sensitivities = {}

for scale in ALL_SCALES:
    sensitivities = []
    
    # Experimento 1
    scale_df1 = df_exp1[df_exp1['Scale'] == scale]
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            config = scale_df1[(scale_df1['Alpha'] == alpha) & (scale_df1['Beta'] == beta)]
            if len(config) >= 2:
                simple = config[config['Model'] == 'Simple']['FIC'].values[0]
                complex = config[config['Model'] == 'Complejo']['FIC'].values[0]
                sensitivities.append(abs(complex - simple))
    
    # Experimento 2
    scale_df2 = df_exp2[df_exp2['Scale'] == scale]
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            config = scale_df2[(scale_df2['Alpha'] == alpha) & (scale_df2['Beta'] == beta)]
            if len(config) >= 2:
                models = config['FIC'].values
                sensitivities.append(np.std(models))
    
    # Experimento 3
    scale_df3 = df_exp3[df_exp3['Scale'] == scale]
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            config = scale_df3[(scale_df3['Alpha'] == alpha) & (scale_df3['Beta'] == beta)]
            if len(config) >= 2:
                models = config['FIC'].values
                sensitivities.append(np.std(models))
    
    if sensitivities:
        scale_sensitivities[scale] = np.mean(sensitivities)

# Ordenar por sensibilidad
sorted_scales = sorted(scale_sensitivities.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Scale':<25} {'Avg Sensitivity':<20} {'Recomendación':<30}")
print("-" * 100)

for i, (scale, sensitivity) in enumerate(sorted_scales, 1):
    if i == 1:
        rec = "⭐ MEJOR - Más sensible"
    elif i <= 3:
        rec = "✅ Buena opción"
    else:
        rec = "⚠️  Menos recomendada"
    
    print(f"{i:<6} {scale:<25} {sensitivity:<20.2f} {rec:<30}")

# ============================================================================
# RECOMENDACIONES FINALES
# ============================================================================

print(f"\n\n{'='*100}")
print("RECOMENDACIONES FINALES PARA TU FIC")
print(f"{'='*100}")

best_scale = sorted_scales[0][0]

print(f"""
🎯 CONFIGURACIÓN RECOMENDADA GENERAL:
   
   Escala: {best_scale}
   Razón: Mayor sensibilidad promedio en todos los experimentos
   
   Coeficientes sugeridos:
   - α (FLOPs): 2.0 - 5.0 (balancea penalización sin dominar)
   - β (params): 0.5 - 1.0 (complementa la penalización por FLOPs)

📋 GUÍA POR CASO DE USO:

1. DEPLOYMENT EN MÓVILES/EDGE (FLOPs críticos):
   - Escala: linear_mega o sqrt_mega
   - α: 5.0 - 10.0 (penaliza fuertemente FLOPs)
   - β: 0.5 (parámetros menos importantes)

2. SELECCIÓN DE MODELOS BALANCEADA:
   - Escala: {best_scale}
   - α: 2.0 (estándar como AIC)
   - β: 1.0 (balance 50/50 FLOPs vs params)

3. OPTIMIZACIÓN DE ARQUITECTURAS:
   - Escala: log_params_ratio (eficiencia por parámetro)
   - α: 2.0 - 5.0
   - β: 1.0 - 2.0

4. DATASETS GRANDES:
   - Usar variante 'bic' (α crece con n)
   - Escala: {best_scale}
   - β: 0.5 - 1.0

💡 PRÓXIMOS PASOS:

1. Exporta los DataFrames (df_exp1, df_exp2, df_exp3) para análisis detallado
2. Visualiza la relación α vs β vs ΔFIC para cada escala
3. Prueba con tus propios datasets y arquitecturas
4. Ajusta α y β según tus prioridades de deployment
""")

print(f"\n{'='*100}")
print("EXPERIMENTOS COMPLETADOS")
print(f"{'='*100}")
print(f"\nResultados guardados en DataFrames:")
print(f"   - df_exp1: Regresión lineal ({len(df_exp1)} configuraciones)")
print(f"   - df_exp2: Redes neuronales ({len(df_exp2)} configuraciones)")
print(f"   - df_exp3: Polinomios ({len(df_exp3)} configuraciones)")
print(f"\nTotal de experimentos ejecutados: {len(df_exp1) + len(df_exp2) + len(df_exp3)}")