"""
FIC (FLOPs Information Criterion) - Ejemplos de Uso

Reconstrucción en formato .py del notebook homónimo (fic_usage.ipynb)
"""

# ===========================================================================
# 1: Configuración inicial e imports
# ===========================================================================
# Configurar paths del proyecto y librerías necesarias
# Incluye fix para warnings de OpenMP en Windows

import sys
import os

project_path = r'C:\Users\hecto\OneDrive\Escritorio\Personal\iroFactory\31.FLOPs-Information-Criterion'
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Fix OpenMP warning en Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import flop_counter
from flop_counter.flop_information_criterion import FlopInformationCriterion

# Configurar seed para reproducibilidad
np.random.seed(42)

print("Configuración completada")
print(f"FlopInformationCriterion importado correctamente")

# ===========================================================================
# 2: Criterios tradicionales de información
# ===========================================================================
# Definir funciones para calcular AIC, BIC, HQIC y MDL
# Estos son los criterios clásicos que NO consideran FLOPs

def calculate_aic(log_likelihood: float, k: int) -> float:
    """
    AIC: Akaike Information Criterion
    AIC = -2*log(L) + 2*k
    Penaliza solo por número de parámetros
    """
    return log_likelihood + 2 * k


def calculate_bic(log_likelihood: float, k: int, n: int) -> float:
    """
    BIC: Bayesian Information Criterion  
    BIC = -2*log(L) + k*log(n)
    Penaliza más que AIC, favorece modelos más simples
    """
    return log_likelihood + k * np.log(n)


def calculate_hqic(log_likelihood: float, k: int, n: int) -> float:
    """
    HQIC: Hannan-Quinn Information Criterion
    HQIC = -2*log(L) + 2*k*log(log(n))
    Penalización intermedia entre AIC y BIC
    """
    return log_likelihood + 2 * k * np.log(np.log(n))


def calculate_mdl(log_likelihood: float, k: int, n: int) -> float:
    """
    MDL: Minimum Description Length
    MDL = -log(L) + (k/2)*log(n)
    Similar a BIC pero con constantes diferentes
    """
    return log_likelihood / 2 + (k / 2) * np.log(n)

print("Criterios tradicionales definidos: AIC, BIC, HQIC, MDL")

# ===========================================================================
# 3: Funciones de visualización y comparación
# ===========================================================================
# Funciones auxiliares para mostrar resultados de forma clara

def print_criteria_comparison(name: str, results: dict, show_all: bool = False):
    """
    Imprime comparación detallada de criterios para un modelo.
    Muestra información del modelo y valores de AIC, BIC, FIC.
    """
    print(f"\n{'='*80}")
    print(f"MODELO: {name}")
    print(f"{'='*80}")
    
    # Información básica del modelo
    print(f"\nInformación del Modelo:")
    print(f"  Parámetros:       {results.get('n_params', 'N/A'):,}")
    print(f"  FLOPs:            {results.get('flops', 'N/A'):,}")
    print(f"  Muestras:         {results.get('n_samples', 'N/A')}")
    print(f"  Log-Likelihood:   {results.get('log_likelihood_term', 'N/A'):.2f}")
    
    if 'accuracy' in results:
        acc_metric = 'R²' if results.get('accuracy', 0) <= 1 else 'Accuracy'
        print(f"  {acc_metric}:            {results['accuracy']:.4f}")
    
    # Valores de criterios de información
    print(f"\nCriterios de Información:")
    print(f"  {'Criterio':<15} {'Valor':<15} {'Penalización':<20}")
    print(f"  {'-'*50}")
    
    if 'aic' in results:
        print(f"  {'AIC':<15} {results['aic']:<15.2f} {'2k':<20}")
    
    if 'bic' in results:
        print(f"  {'BIC':<15} {results['bic']:<15.2f} {'k*log(n)':<20}")
    
    # FIC destacado
    lambda_val = results.get('lambda', 'N/A')
    print(f"  {'FIC':<15} {results['fic']:<15.2f} {'α*f(FLOPs) + β*k':<20} <- Incluye FLOPs")
    
    # Criterios adicionales opcionales
    if show_all:
        if 'hqic' in results:
            print(f"  {'HQIC':<15} {results['hqic']:<15.2f} {'2k*log(log(n))':<20}")
        
        if 'mdl' in results:
            print(f"  {'MDL':<15} {results['mdl']:<15.2f} {'(k/2)*log(n)':<20}")
    
    # Desglose detallado del FIC
    print(f"\nDesglose del FIC:")
    print(f"  Ajuste (likelihood):       {results['log_likelihood_term']:.2f}")
    print(f"  Penalización FLOPs:        {results['flops_penalty']:.2f}")
    print(f"  Penalización parámetros:   {results['params_penalty']:.2f}")
    print(f"  Coeficientes: α={results['alpha']:.2f}, β={results['beta']:.2f}, λ={lambda_val}")


def compare_all_criteria(models_results: dict, show_all: bool = False):
    """
    Compara todos los modelos lado a lado según diferentes criterios.
    Identifica el mejor modelo según cada criterio y calcula diferencias ΔFIC.
    """
    print(f"\n{'='*80}")
    print("COMPARACIÓN DE MODELOS")
    print(f"{'='*80}")
    
    model_names = list(models_results.keys())
    criteria = ['AIC', 'BIC', 'FIC']
    
    if show_all:
        criteria.extend(['HQIC', 'MDL'])
    
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
    
    # Identificar mejor modelo según cada criterio
    print(f"\n{'Criterio':<15} {'Mejor Modelo':<20} {'Valor':<15}")
    print("-" * 50)
    
    for criterion in criteria:
        key = criterion.lower()
        best_name = min(model_names, key=lambda x: models_results[x].get(key, float('inf')))
        best_value = models_results[best_name][key]
        print(f"{criterion:<15} {best_name:<20} {best_value:<15.2f}")
    
    # Análisis de diferencias ΔFIC
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
        
        # Interpretación de la magnitud de diferencia
        if delta_fic < 2:
            interpretation = "Equivalente al mejor"
        elif delta_fic < 10:
            interpretation = "Evidencia sustancial contra"
        else:
            interpretation = "Evidencia fuerte contra"
        
        marker = "* MEJOR" if delta_fic < 0.01 else ""
        
        print(f"{name:<20} {fic:<15.2f} {delta_fic:<15.2f} {interpretation:<30} {marker}")

print("Funciones de visualización definidas")

# ===========================================================================
# 4: Ejemplo 1 - Regresión Lineal Simple vs Compleja
# ===========================================================================
# Demostración: Dos modelos con igual número de parámetros pero diferentes FLOPs
# Objetivo: Mostrar que AIC/BIC son ciegos a eficiencia computacional

print("\n" + "="*80)
print("EJEMPLO 1: REGRESIÓN LINEAL")
print("="*80)
print("\nEscenario: Dos modelos lineales con igual # de parámetros")
print("           pero diferente complejidad computacional")

# Generar datos sintéticos de regresión lineal
n_samples = 100
X_train = np.random.randn(n_samples, 1)
true_slope = 2.5
true_intercept = 1.0
noise = np.random.randn(n_samples) * 0.5
y_train = true_slope * X_train.squeeze() + true_intercept + noise

def linear_model_simple(X):
    """
    Modelo lineal eficiente: y = ax + b
    Implementación directa sin operaciones innecesarias
    """
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    return X_extended @ W

def linear_model_complex(X):
    """
    Modelo lineal ineficiente: y = ax + b
    Mismo resultado pero con operaciones costosas innecesarias
    Simula código mal optimizado o arquitectura redundante
    """
    W = np.array([[true_slope], [true_intercept]])
    X_extended = np.column_stack([X, np.ones(len(X))])
    
    # Operaciones adicionales que no aportan valor
    temp = X_extended @ W
    temp = np.exp(np.log(temp))  # Identidad costosa: exp(log(x)) = x
    temp = temp @ np.eye(1)      # Multiplicación innecesaria por identidad
    
    return temp

print("Modelos definidos:")
print("  - Simple: Implementación directa (~100 FLOPs)")
print("  - Complejo: Con operaciones innecesarias (~500 FLOPs)")

# ===========================================================================
# 5: Evaluar modelos de regresión con FIC
# ===========================================================================
# Crear calculador de FIC y evaluar ambos modelos
# Configuración: α=5.0 (penaliza FLOPs), β=0.5 (menos peso a params), λ=0.3

# Inicializar calculador con configuración balanceada
fic_calc = FlopInformationCriterion(
    variant='custom',
    alpha=5.0,      # Penalización fuerte de FLOPs
    beta=0.5,       # Menos peso a parámetros
    lambda_balance=0.3  # 30% log, 70% lineal
)

print("Evaluando modelo simple...")
result_simple = fic_calc.evaluate_model(
    model=linear_model_simple,
    X=X_train,
    y_true=y_train,
    task='regression',
    n_params=2,
    framework='numpy'
)

print("Evaluando modelo complejo...")
result_complex = fic_calc.evaluate_model(
    model=linear_model_complex,
    X=X_train,
    y_true=y_train,
    task='regression',
    n_params=2,
    framework='numpy'
)

print("Evaluación completada")

# ===========================================================================
# 6: Calcular criterios tradicionales y comparar
# ===========================================================================
# Agregar AIC, BIC, HQIC, MDL a los resultados y mostrar comparación

# Calcular criterios tradicionales para ambos modelos
for name, result in [("Simple", result_simple), ("Complejo", result_complex)]:
    result['aic'] = calculate_aic(result['log_likelihood_term'], result['n_params'])
    result['bic'] = calculate_bic(result['log_likelihood_term'], result['n_params'], result['n_samples'])
    result['hqic'] = calculate_hqic(result['log_likelihood_term'], result['n_params'], result['n_samples'])
    result['mdl'] = calculate_mdl(result['log_likelihood_term'], result['n_params'], result['n_samples'])

# Mostrar resultados individuales
print_criteria_comparison("Modelo Simple", result_simple, show_all=True)
print_criteria_comparison("Modelo Complejo", result_complex, show_all=True)

# Comparación lado a lado
models_linear = {
    'Simple': result_simple,
    'Complejo': result_complex
}
compare_all_criteria(models_linear, show_all=True)

# ===========================================================================
# 7: Ejemplo 2 - Redes Neuronales con diferentes arquitecturas
# ===========================================================================
# Demostración: Comparar arquitecturas Wide vs Deep vs Balanced
# Objetivo: Mostrar que profundidad y ancho afectan FLOPs diferente

print("\n" + "="*80)
print("EJEMPLO 2: REDES NEURONALES - ARQUITECTURAS")
print("="*80)
print("\nEscenario: Tres arquitecturas para clasificación multiclase")
print("           Diferentes trade-offs entre ancho y profundidad")

# Generar datos de clasificación
n_samples = 200
n_features = 20
n_classes = 3

X_train = np.random.randn(n_samples, n_features)
y_train = np.random.randint(0, n_classes, n_samples)
y_train_onehot = np.eye(n_classes)[y_train]

def softmax(x):
    """Softmax estable numéricamente para evitar overflow"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

print("Datos de clasificación generados:")
print(f"  Muestras: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Clases: {n_classes}")

# ===========================================================================
# 8: Definir arquitecturas de redes neuronales
# ===========================================================================
# Tres diseños diferentes: ancha-poco profunda, profunda-estrecha, balanceada

def wide_shallow_net(X):
    """
    Red ancha y poco profunda: 20 -> 100 -> 3
    Características:
      - Pocas capas (2)
      - Muchas neuronas por capa (100 hidden)
      - ~2,303 parámetros
      - FLOPs relativamente bajos (pocas operaciones secuenciales)
    """
    W1 = np.random.randn(20, 100) * 0.01
    b1 = np.zeros(100)
    W2 = np.random.randn(100, 3) * 0.01
    b2 = np.zeros(3)
    
    h1 = np.maximum(0, X @ W1 + b1)  # ReLU
    logits = h1 @ W2 + b2
    return softmax(logits)

def deep_narrow_net(X):
    """
    Red profunda y estrecha: 20 -> 30 -> 30 -> 30 -> 3
    Características:
      - Muchas capas (4)
      - Pocas neuronas por capa (30 hidden)
      - ~2,523 parámetros
      - FLOPs altos (muchas operaciones secuenciales)
    """
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
    """
    Red balanceada: 20 -> 50 -> 25 -> 3
    Características:
      - Capas intermedias (3)
      - Neuronas intermedias (50, 25)
      - ~2,353 parámetros
      - FLOPs medios (balance profundidad/ancho)
    """
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

# Calcular número exacto de parámetros
n_params_wide = (20*100 + 100) + (100*3 + 3)  # 2,303
n_params_deep = (20*30 + 30) + (30*30 + 30) + (30*30 + 30) + (30*3 + 3)  # 2,523
n_params_balanced = (20*50 + 50) + (50*25 + 25) + (25*3 + 3)  # 2,353

print("Arquitecturas definidas:")
print(f"  Wide-Shallow: {n_params_wide:,} parámetros")
print(f"  Deep-Narrow:  {n_params_deep:,} parámetros")
print(f"  Balanced:     {n_params_balanced:,} parámetros")

# ===========================================================================
# 9: Evaluar y comparar redes neuronales
# ===========================================================================
# Evaluar las tres arquitecturas y compararlas con todos los criterios

results_nn = {}

for name, model, n_params in [
    ("Wide-Shallow", wide_shallow_net, n_params_wide),
    ("Deep-Narrow", deep_narrow_net, n_params_deep),
    ("Balanced", balanced_net, n_params_balanced)
]:
    print(f"\nEvaluando: {name}...")
    
    result = fic_calc.evaluate_model(
        model=model,
        X=X_train,
        y_true=y_train,
        task='classification',
        n_params=n_params,
        framework='numpy'
    )
    
    # Agregar criterios tradicionales
    result['aic'] = calculate_aic(result['log_likelihood_term'], n_params)
    result['bic'] = calculate_bic(result['log_likelihood_term'], n_params, n_samples)
    result['hqic'] = calculate_hqic(result['log_likelihood_term'], n_params, n_samples)
    result['mdl'] = calculate_mdl(result['log_likelihood_term'], n_params, n_samples)
    
    results_nn[name] = result
    
    print_criteria_comparison(name, result, show_all=False)

# Comparación completa
compare_all_criteria(results_nn, show_all=False)

# ===========================================================================
# 10: Ejemplo 3 - CNNs en PyTorch
# ===========================================================================
# Demostración: Redes convolucionales simples vs profundas
# Objetivo: Mostrar FIC en modelos de PyTorch reales

print("\n" + "="*80)
print("EJEMPLO 3: REDES CONVOLUCIONALES (PyTorch)")
print("="*80)
print("\nEscenario: CNNs con diferente profundidad")
print("           Comparar costo de convoluciones")

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    CNN simple: 2 capas convolucionales + clasificador
    Arquitectura: Conv(3->16) -> Pool -> Conv(16->32) -> Pool -> FC(2048->10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

class DeepCNN(nn.Module):
    """
    CNN profunda: 4 capas convolucionales + clasificador
    Arquitectura: Conv(3->16) -> Pool -> Conv(16->32) -> Pool -> 
                  Conv(32->64) -> Pool -> Conv(64->64) -> Pool -> FC(256->10)
    Más profunda = Más FLOPs
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 2 * 2, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

# Datos dummy para evaluación
X_torch = torch.randn(16, 3, 32, 32)  # 16 imágenes 32x32 RGB
y_torch = torch.randint(0, 10, (16,))  # 10 clases

simple_cnn = SimpleCNN()
deep_cnn = DeepCNN()

print("CNNs definidas:")
print(f"  SimpleCNN: 2 capas conv + 1 FC")
print(f"  DeepCNN:   4 capas conv + 1 FC")

# ===========================================================================
# 11: Evaluar CNNs de PyTorch
# ===========================================================================
# Evaluar ambas CNNs con FIC y mostrar diferencias

results_torch = {}

for name, model in [("SimpleCNN", simple_cnn), ("DeepCNN", deep_cnn)]:
    print(f"\nEvaluando: {name}...")
    
    result = fic_calc.evaluate_model(
        model=model,
        X=X_torch,
        y_true=y_torch.numpy(),
        task='classification',
        framework='torch'
    )
    
    # Agregar criterios tradicionales
    n_params = result['n_params']
    result['aic'] = calculate_aic(result['log_likelihood_term'], n_params)
    result['bic'] = calculate_bic(result['log_likelihood_term'], n_params, 16)
    
    results_torch[name] = result
    print_criteria_comparison(name, result, show_all=False)

compare_all_criteria(results_torch, show_all=False)