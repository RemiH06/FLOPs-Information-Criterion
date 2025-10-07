"""
FIC Parameter Optimization
==========================
Experimentación sistemática para encontrar los valores óptimos de:
- alpha: peso de penalización de FLOPs
- beta: peso de penalización de parámetros  
- lambda: balance entre término logarítmico y lineal

"""

import sys
import os
import numpy as np
import pandas as pd
from flop_counter.flop_information_criterion import FlopInformationCriterion

# Configuración de reproducibilidad
np.random.seed(42)

# ===========================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ===========================================================================

# Rango de valores a explorar para lambda (balance log vs linear)
# lambda=0.0: 100% lineal (diferencias absolutas en FLOPs)
# lambda=1.0: 100% logarítmico (diferencias relativas en FLOPs)
LAMBDA_VALUES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

# Valores de alpha a probar (penalización de FLOPs)
# Valores más altos penalizan más fuertemente los FLOPs
ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0, 7.0, 10.0]

# Valores de beta a probar (penalización de parámetros)
# Valores más altos penalizan más fuertemente el número de parámetros
BETA_VALUES = [0.0, 0.3, 0.5, 1.0, 2.0]

print("="*100)
print("FIC: OPTIMIZACIÓN DE PARÁMETROS")
print("="*100)
print(f"\nConfiguración:")
print(f"  Valores de lambda: {len(LAMBDA_VALUES)}")
print(f"  Valores de alpha: {len(ALPHA_VALUES)}")
print(f"  Valores de beta: {len(BETA_VALUES)}")
print(f"  Total configuraciones por experimento: {len(LAMBDA_VALUES) * len(ALPHA_VALUES) * len(BETA_VALUES)}")

# ===========================================================================
# FUNCIONES AUXILIARES
# ===========================================================================

def calculate_aic(log_likelihood: float, k: int) -> float:
    """
    Calcula el criterio AIC (Akaike Information Criterion).
    AIC = -2*log(L) + 2*k
    """
    return log_likelihood + 2 * k


def calculate_bic(log_likelihood: float, k: int, n: int) -> float:
    """
    Calcula el criterio BIC (Bayesian Information Criterion).
    BIC = -2*log(L) + k*log(n)
    """
    return log_likelihood + k * np.log(n)


def create_results_dataframe(results_dict):
    """
    Convierte el diccionario de resultados a un DataFrame de pandas
    para facilitar el análisis.
    
    Args:
        results_dict: Diccionario con estructura {(lambda, alpha, beta, model): result}
    
    Returns:
        DataFrame con todas las métricas organizadas
    """
    data = []
    for key, result in results_dict.items():
        lambda_val, alpha, beta, model = key
        
        data.append({
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


def analyze_sensitivity(df, experiment_name):
    """
    Analiza cómo cada combinación de parámetros afecta la capacidad
    del FIC para distinguir entre modelos.
    
    La sensibilidad se mide como la diferencia promedio en FIC entre
    el modelo de referencia y los demás modelos.
    
    Args:
        df: DataFrame con resultados del experimento
        experiment_name: Nombre del experimento para el reporte
    
    Returns:
        Lista de estadísticas por cada valor de lambda
    """
    print(f"\n{'='*100}")
    print(f"ANÁLISIS DE SENSIBILIDAD: {experiment_name}")
    print(f"{'='*100}")
    
    # Determinar modelo de referencia (el primero alfabéticamente)
    all_models = df['Model'].unique()
    reference_model = sorted(all_models)[0]
    
    print(f"\nModelo de referencia: {reference_model}")
    print(f"\n{'Lambda':<10} {'Alpha':<10} {'Beta':<10} {'Avg DELTA_FIC':<20} {'Interpretation':<30}")
    print("-" * 80)
    
    best_configs = []
    
    # Analizar cada combinación de lambda, alpha, beta
    for lambda_val in sorted(df['Lambda'].unique()):
        for alpha in sorted(df['Alpha'].unique()):
            for beta in sorted(df['Beta'].unique()):
                # Filtrar configuración específica
                config_df = df[
                    (df['Lambda'] == lambda_val) & 
                    (df['Alpha'] == alpha) & 
                    (df['Beta'] == beta)
                ]
                
                if len(config_df) < 2:
                    continue
                
                # Calcular diferencia promedio respecto al modelo de referencia
                ref_fic = config_df[config_df['Model'] == reference_model]['FIC'].values
                other_fics = config_df[config_df['Model'] != reference_model]['FIC'].values
                
                if len(ref_fic) > 0 and len(other_fics) > 0:
                    avg_delta = np.mean(np.abs(other_fics - ref_fic[0]))
                    
                    # Interpretación de la sensibilidad
                    if avg_delta < 1:
                        interp = "Muy baja - no distingue"
                    elif avg_delta < 10:
                        interp = "Baja"
                    elif avg_delta < 50:
                        interp = "Media"
                    elif avg_delta < 100:
                        interp = "Alta"
                    else:
                        interp = "Muy alta - buena distinción"
                    
                    best_configs.append({
                        'lambda': lambda_val,
                        'alpha': alpha,
                        'beta': beta,
                        'avg_delta': avg_delta,
                        'interpretation': interp
                    })
                    
                    print(f"{lambda_val:<10.1f} {alpha:<10.1f} {beta:<10.1f} {avg_delta:<20.2f} {interp:<30}")
    
    # Encontrar mejor configuración
    if best_configs:
        best = max(best_configs, key=lambda x: x['avg_delta'])
        print(f"\n--- MEJOR CONFIGURACIÓN PARA ESTE EXPERIMENTO ---")
        print(f"Lambda = {best['lambda']:.1f}, Alpha = {best['alpha']:.1f}, Beta = {best['beta']:.1f}")
        print(f"Sensibilidad promedio: {best['avg_delta']:.2f}")
        print(f"Interpretación: {best['interpretation']}")
    
    return best_configs


def print_experiment_header(title, models_info):
    """
    Imprime el encabezado informativo de cada experimento.
    
    Args:
        title: Título del experimento
        models_info: Dict con información de cada modelo
    """
    print(f"\n\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    print("\nModelos a comparar:")
    for model_name, info in models_info.items():
        print(f"   {model_name}: {info['params']} params, {info['expected_flops']} FLOPs esperados")


# ===========================================================================
# EXPERIMENTO 1: REGRESIÓN LINEAL
# ===========================================================================
# Objetivo: Detectar modelos con igual número de parámetros pero diferentes FLOPs
# Caso: Modelo simple vs modelo con operaciones innecesarias

print_experiment_header(
    "EXPERIMENTO 1: REGRESIÓN LINEAL - EFICIENCIA COMPUTACIONAL",
    {
        'Simple': {'params': 2, 'expected_flops': '~500'},
        'Ineficiente': {'params': 2, 'expected_flops': '~5000 (10x más)'}
    }
)

# Generar datos de regresión lineal
n_samples = 100
X_train = np.random.randn(n_samples, 1)
true_slope = 2.5
true_intercept = 1.0
noise = np.random.randn(n_samples) * 0.5
y_train = true_slope * X_train.squeeze() + true_intercept + noise


def linear_model_simple(X):
    """Modelo lineal eficiente: y = ax + b"""
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    return X_extended @ W


def linear_model_inefficient(X):
    """
    Modelo lineal con operaciones costosas innecesarias.
    Produce el mismo resultado que el modelo simple pero con ~10x más FLOPs.
    """
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    temp = X_extended @ W
    
    # Operaciones costosas que no aportan valor
    for _ in range(5):
        temp = np.exp(np.log(np.abs(temp) + 1e-10)) * np.sign(temp)
        temp = temp @ np.eye(1)
        temp = np.sin(np.arcsin(np.clip(
            temp / (np.max(np.abs(temp)) + 1e-10), -0.99, 0.99
        ))) * (np.max(np.abs(temp)) + 1e-10)
    
    return temp


print(f"\nEjecutando experimento 1...")
print(f"Total de configuraciones a probar: {len(LAMBDA_VALUES) * len(ALPHA_VALUES) * len(BETA_VALUES)}")

results_exp1 = {}

for lambda_val in LAMBDA_VALUES:
    print(f"   Procesando lambda = {lambda_val:.1f}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            # Crear calculador de FIC con esta configuración
            fic_calc = FlopInformationCriterion(
                variant='custom',
                alpha=alpha,
                beta=beta,
                lambda_balance=lambda_val
            )
            
            # Evaluar ambos modelos
            try:
                result_simple = fic_calc.evaluate_model(
                    model=linear_model_simple,
                    X=X_train,
                    y_true=y_train,
                    task='regression',
                    n_params=2,
                    framework='numpy'
                )
                
                result_inefficient = fic_calc.evaluate_model(
                    model=linear_model_inefficient,
                    X=X_train,
                    y_true=y_train,
                    task='regression',
                    n_params=2,
                    framework='numpy'
                )
                
                results_exp1[(lambda_val, alpha, beta, 'Simple')] = result_simple
                results_exp1[(lambda_val, alpha, beta, 'Ineficiente')] = result_inefficient
                
            except Exception as e:
                print(f"      Error en lambda={lambda_val}, alpha={alpha}, beta={beta}: {e}")
                continue

# Analizar resultados
df_exp1 = create_results_dataframe(results_exp1)
configs_exp1 = analyze_sensitivity(df_exp1, "Experimento 1: Regresión Lineal")


# ===========================================================================
# EXPERIMENTO 2: REDES NEURONALES
# ===========================================================================
# Objetivo: Comparar arquitecturas con diferente profundidad y ancho
# Caso: Wide vs Deep vs Balanced

print_experiment_header(
    "EXPERIMENTO 2: REDES NEURONALES - ARQUITECTURAS",
    {
        'Wide-Shallow': {'params': 2303, 'expected_flops': 'bajo'},
        'Deep-Narrow': {'params': 2523, 'expected_flops': 'alto'},
        'Balanced': {'params': 2353, 'expected_flops': 'medio'}
    }
)

# Generar datos de clasificación
n_samples = 200
n_features = 20
n_classes = 3
X_train = np.random.randn(n_samples, n_features)
y_train = np.random.randint(0, n_classes, n_samples)


def softmax(x):
    """Función softmax estable numéricamente"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def wide_shallow_net(X):
    """Red ancha y poco profunda: 20 -> 100 -> 3"""
    W1 = np.random.randn(20, 100) * 0.01
    b1 = np.zeros(100)
    W2 = np.random.randn(100, 3) * 0.01
    b2 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)
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


# Calcular número de parámetros
n_params_wide = (20*100 + 100) + (100*3 + 3)
n_params_deep = (20*30 + 30) + (30*30 + 30) + (30*30 + 30) + (30*3 + 3)
n_params_balanced = (20*50 + 50) + (50*25 + 25) + (25*3 + 3)

print(f"\nEjecutando experimento 2...")

results_exp2 = {}

for lambda_val in LAMBDA_VALUES:
    print(f"   Procesando lambda = {lambda_val:.1f}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            fic_calc = FlopInformationCriterion(
                variant='custom',
                alpha=alpha,
                beta=beta,
                lambda_balance=lambda_val
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
                        framework='numpy'
                    )
                    
                    results_exp2[(lambda_val, alpha, beta, name)] = result
                    
                except Exception as e:
                    print(f"      Error en {name}: {e}")
                    continue

# Analizar resultados
df_exp2 = create_results_dataframe(results_exp2)
configs_exp2 = analyze_sensitivity(df_exp2, "Experimento 2: Redes Neuronales")


# ===========================================================================
# EXPERIMENTO 3: REGRESIÓN POLINOMIAL
# ===========================================================================
# Objetivo: Detectar overfitting y complejidad innecesaria
# Caso: Polinomios de diferentes grados (el óptimo es grado 2)

print_experiment_header(
    "EXPERIMENTO 3: REGRESIÓN POLINOMIAL - OVERFITTING",
    {
        'Linear': {'params': 2, 'expected_flops': 'muy bajo'},
        'Quadratic': {'params': 3, 'expected_flops': 'bajo (óptimo)'},
        'Cubic': {'params': 4, 'expected_flops': 'medio'},
        'Poly5': {'params': 6, 'expected_flops': 'alto'},
        'Poly10': {'params': 11, 'expected_flops': 'muy alto'}
    }
)

# Generar datos con relación cuadrática verdadera
n_samples = 150
X_poly = np.random.randn(n_samples, 1)
y_poly = 2.0 * X_poly.squeeze()**2 + 1.5 * X_poly.squeeze() + 1.0
y_poly += np.random.randn(n_samples) * 0.5


def create_poly_model(degree):
    """
    Crea un modelo polinomial de grado especificado.
    
    Args:
        degree: Grado del polinomio
    
    Returns:
        Función que implementa el modelo polinomial
    """
    def model(X_input):
        # Crear features polinomiales
        X_features = np.column_stack([X_input**i for i in range(degree + 1)])
        
        # Coeficientes con valores verdaderos para términos relevantes
        coeffs = np.random.randn(degree + 1) * 0.1
        coeffs[0] = 1.0      # intercept
        if degree >= 1:
            coeffs[1] = 1.5  # término lineal
        if degree >= 2:
            coeffs[2] = 2.0  # término cuadrático (correcto)
        
        return X_features @ coeffs
    
    return model


print(f"\nEjecutando experimento 3...")

results_exp3 = {}

for lambda_val in LAMBDA_VALUES:
    print(f"   Procesando lambda = {lambda_val:.1f}")
    
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            fic_calc = FlopInformationCriterion(
                variant='custom',
                alpha=alpha,
                beta=beta,
                lambda_balance=lambda_val
            )
            
            for degree, name in [(1, 'Linear'), (2, 'Quadratic'), (3, 'Cubic'),
                                  (5, 'Poly5'), (10, 'Poly10')]:
                try:
                    model = create_poly_model(degree)
                    
                    result = fic_calc.evaluate_model(
                        model=model,
                        X=X_poly,
                        y_true=y_poly,
                        task='regression',
                        n_params=degree + 1,
                        framework='numpy'
                    )
                    
                    results_exp3[(lambda_val, alpha, beta, name)] = result
                    
                except Exception as e:
                    print(f"      Error en {name}: {e}")
                    continue

# Analizar resultados
df_exp3 = create_results_dataframe(results_exp3)
configs_exp3 = analyze_sensitivity(df_exp3, "Experimento 3: Regresión Polinomial")


# ===========================================================================
# ANÁLISIS GLOBAL Y RECOMENDACIONES
# ===========================================================================

print(f"\n\n{'='*100}")
print("ANÁLISIS GLOBAL: MEJORES CONFIGURACIONES")
print(f"{'='*100}")

# Encontrar la mejor configuración en cada experimento
if configs_exp1 and configs_exp2 and configs_exp3:
    best_exp1 = max(configs_exp1, key=lambda x: x['avg_delta'])
    best_exp2 = max(configs_exp2, key=lambda x: x['avg_delta'])
    best_exp3 = max(configs_exp3, key=lambda x: x['avg_delta'])
    
    print("\nMejor configuración por experimento:")
    print(f"\nExperimento 1 (Regresión):")
    print(f"   Lambda = {best_exp1['lambda']:.1f}, Alpha = {best_exp1['alpha']:.1f}, Beta = {best_exp1['beta']:.1f}")
    print(f"   Sensibilidad: {best_exp1['avg_delta']:.2f}")
    
    print(f"\nExperimento 2 (Redes Neuronales):")
    print(f"   Lambda = {best_exp2['lambda']:.1f}, Alpha = {best_exp2['alpha']:.1f}, Beta = {best_exp2['beta']:.1f}")
    print(f"   Sensibilidad: {best_exp2['avg_delta']:.2f}")
    
    print(f"\nExperimento 3 (Polinomios):")
    print(f"   Lambda = {best_exp3['lambda']:.1f}, Alpha = {best_exp3['alpha']:.1f}, Beta = {best_exp3['beta']:.1f}")
    print(f"   Sensibilidad: {best_exp3['avg_delta']:.2f}")
    
    # Calcular configuración promedio recomendada
    avg_lambda = np.mean([best_exp1['lambda'], best_exp2['lambda'], best_exp3['lambda']])
    avg_alpha = np.mean([best_exp1['alpha'], best_exp2['alpha'], best_exp3['alpha']])
    avg_beta = np.mean([best_exp1['beta'], best_exp2['beta'], best_exp3['beta']])
    
    print(f"\n{'='*100}")
    print("CONFIGURACIÓN RECOMENDADA (PROMEDIO)")
    print(f"{'='*100}")
    print(f"\nLambda = {avg_lambda:.2f}")
    print(f"Alpha  = {avg_alpha:.2f}")
    print(f"Beta   = {avg_beta:.2f}")
    
    print(f"\nFórmula del FIC con estos valores:")
    print(f"FIC = -2*log(L) + {avg_alpha:.1f} * [{avg_lambda:.1f}*log(FLOPs) + {1-avg_lambda:.1f}*FLOPs/1e6] + {avg_beta:.1f}*k")
    
    print(f"\nInterpretación:")
    if avg_lambda < 0.3:
        print("   Lambda bajo: El término lineal domina (sensible a diferencias absolutas)")
    elif avg_lambda > 0.7:
        print("   Lambda alto: El término logarítmico domina (sensible a diferencias relativas)")
    else:
        print("   Lambda balanceado: Combina sensibilidad relativa y absoluta")
    
    if avg_alpha > 5:
        print("   Alpha alto: Penalización fuerte de FLOPs (prioriza eficiencia)")
    else:
        print("   Alpha moderado: Balance entre precisión y eficiencia")
    
    if avg_beta < 0.5:
        print("   Beta bajo: Poco peso a parámetros (FLOPs dominan)")
    elif avg_beta > 1:
        print("   Beta alto: Peso significativo a parámetros")
    else:
        print("   Beta moderado: Balance entre FLOPs y parámetros")

print(f"\n{'='*100}")
print("OPTIMIZACIÓN COMPLETADA")
print(f"{'='*100}")
print(f"\nDataFrames disponibles para análisis adicional:")
print(f"   - df_exp1: Experimento de regresión lineal")
print(f"   - df_exp2: Experimento de redes neuronales")
print(f"   - df_exp3: Experimento de regresión polinomial")