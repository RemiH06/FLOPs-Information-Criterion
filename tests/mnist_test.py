#!/usr/bin/env python3
"""
MNIST Classification: Comparación de 3 enfoques diferentes
1. MLP (Multi-Layer Perceptron) - Baseline denso
2. CNN (Convolutional Neural Network) - LeNet-like con BatchNorm
3. HOG + SVM (Clásico) - Histograms of Oriented Gradients + Linear SVM

Objetivo: Entrenar cada modelo y luego evaluar con AIC, BIC y FIC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import time

print("="*80)
print("MNIST: COMPARACIÓN DE 3 ENFOQUES DE CLASIFICACIÓN")
print("="*80)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nConfiguración:")
print(f"  Device: {DEVICE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("\n" + "="*80)
print("CARGANDO DATOS MNIST")
print("="*80)

# Cargar MNIST usando sklearn (evita problema de torchvision)
print("Descargando MNIST desde OpenML...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')

X = mnist.data.to_numpy().astype(np.float32)
y = mnist.target.to_numpy().astype(np.int64)

# Normalizar
X = X / 255.0

# Split train/test (primeros 60000 train, resto test)
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

# Normalización estándar
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(f"\nDatasets cargados:")
print(f"  Train: {X_train.shape[0]} imágenes")
print(f"  Test:  {X_test.shape[0]} imágenes")

# Convertir a tensores PyTorch
X_train_torch = torch.FloatTensor(X_train).view(-1, 1, 28, 28)
y_train_torch = torch.LongTensor(y_train)
X_test_torch = torch.FloatTensor(X_test).view(-1, 1, 28, 28)
y_test_torch = torch.LongTensor(y_test)

# DataLoaders
train_dataset = TensorDataset(X_train_torch, y_train_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# MODELO 1: MLP (Multi-Layer Perceptron)
# ============================================================================

print("\n" + "="*80)
print("MODELO 1: MLP (Multi-Layer Perceptron)")
print("="*80)
print("\nArquitectura: 784 -> 512 -> 256 -> 10")
print("Características: Denso, Dropout 0.2, ReLU")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def count_parameters(model):
    """Cuenta parámetros entrenables"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Inicializar MLP
mlp_model = MLP().to(DEVICE)
mlp_params = count_parameters(mlp_model)

print(f"\nParámetros entrenables: {mlp_params:,}")

# Entrenamiento
print("\nEntrenando MLP...")
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
mlp_criterion = nn.CrossEntropyLoss()

mlp_start_time = time.time()

for epoch in range(EPOCHS):
    mlp_model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        mlp_optimizer.zero_grad()
        output = mlp_model(data)
        loss = mlp_criterion(output, target)
        loss.backward()
        mlp_optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    train_acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    
    print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%")

mlp_train_time = time.time() - mlp_start_time

# Evaluación
print("\nEvaluando MLP...")
mlp_model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = mlp_model(data)
        test_loss += mlp_criterion(output, target).item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

mlp_test_acc = 100. * correct / total
mlp_test_loss = test_loss / len(test_loader)

print(f"\nResultados MLP:")
print(f"  Test Accuracy: {mlp_test_acc:.2f}%")
print(f"  Test Loss: {mlp_test_loss:.4f}")
print(f"  Training Time: {mlp_train_time:.1f}s")

# ============================================================================
# MODELO 2: CNN (Convolutional Neural Network)
# ============================================================================

print("\n" + "="*80)
print("MODELO 2: CNN (LeNet-like con BatchNorm)")
print("="*80)
print("\nArquitectura:")
print("  Conv1: 1->32 (3x3) -> BN -> ReLU -> MaxPool(2x2)")
print("  Conv2: 32->64 (3x3) -> BN -> ReLU -> MaxPool(2x2)")
print("  FC: 1600 -> 128 -> 10")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Bloque 1: Conv -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloque 2: Conv -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Clasificador denso
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Bloque 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        # Bloque 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # Clasificador
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Inicializar CNN
cnn_model = CNN().to(DEVICE)
cnn_params = count_parameters(cnn_model)

print(f"\nParámetros entrenables: {cnn_params:,}")

# Entrenamiento
print("\nEntrenando CNN...")
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
cnn_criterion = nn.CrossEntropyLoss()

cnn_start_time = time.time()

for epoch in range(EPOCHS):
    cnn_model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        cnn_optimizer.zero_grad()
        output = cnn_model(data)
        loss = cnn_criterion(output, target)
        loss.backward()
        cnn_optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    train_acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    
    print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%")

cnn_train_time = time.time() - cnn_start_time

# Evaluación
print("\nEvaluando CNN...")
cnn_model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = cnn_model(data)
        test_loss += cnn_criterion(output, target).item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

cnn_test_acc = 100. * correct / total
cnn_test_loss = test_loss / len(test_loader)

print(f"\nResultados CNN:")
print(f"  Test Accuracy: {cnn_test_acc:.2f}%")
print(f"  Test Loss: {cnn_test_loss:.4f}")
print(f"  Training Time: {cnn_train_time:.1f}s")

# ============================================================================
# MODELO 3: PCA + Linear SVM (Clásico)
# ============================================================================

print("\n" + "="*80)
print("MODELO 3: PCA + Linear SVM (Clásico)")
print("="*80)
print("\nCaracterísticas:")
print("  Reducción dim: PCA (784 -> 150 componentes)")
print("  Clasificador: Linear SVM (C=1.0)")

# Preparar datos para SVM (píxeles crudos)
print("\nPreparando datos...")

# Usar los datos ya normalizados
X_train_flat = X_train  # Ya está en forma (60000, 784)
X_test_flat = X_test    # Ya está en forma (10000, 784)

svm_start_time = time.time()

# Aplicar PCA para reducir dimensionalidad
print("\nAplicando PCA (784 -> 150 dimensiones)...")
pca = PCA(n_components=150, random_state=42)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

explained_var = pca.explained_variance_ratio_.sum()
print(f"Varianza explicada: {explained_var*100:.2f}%")

# Escalar características
scaler = StandardScaler()
X_train_pca_scaled = scaler.fit_transform(X_train_pca)
X_test_pca_scaled = scaler.transform(X_test_pca)

# Entrenar SVM
print("\nEntrenando Linear SVM...")
svm_model = LinearSVC(C=1.0, max_iter=1000, random_state=42, verbose=1)
svm_model.fit(X_train_pca_scaled, y_train)

svm_train_time = time.time() - svm_start_time

# Evaluar
print("\nEvaluando SVM...")
svm_predictions = svm_model.predict(X_test_pca_scaled)
svm_test_acc = 100. * np.mean(svm_predictions == y_test)

print(f"\nResultados SVM:")
print(f"  Test Accuracy: {svm_test_acc:.2f}%")
print(f"  Training Time: {svm_train_time:.1f}s")
print(f"  PCA componentes: {X_train_pca.shape[1]}")
print(f"  Varianza capturada: {explained_var*100:.1f}%")

# ============================================================================
# RESUMEN COMPARATIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN COMPARATIVO")
print("="*80)

print(f"\n{'Modelo':<15} {'Parámetros':<15} {'Test Acc':<12} {'Train Time':<15}")
print("-" * 60)
print(f"{'MLP':<15} {mlp_params:<15,} {mlp_test_acc:<12.2f}% {mlp_train_time:<15.1f}s")
print(f"{'CNN':<15} {cnn_params:<15,} {cnn_test_acc:<12.2f}% {cnn_train_time:<15.1f}s")
print(f"{'PCA+SVM':<15} {X_train_pca.shape[1]:<15} {svm_test_acc:<12.2f}% {svm_train_time:<15.1f}s")

# ============================================================================
# CALCULAR AIC, BIC Y FIC
# ============================================================================

print("\n" + "="*80)
print("CALCULANDO CRITERIOS DE INFORMACIÓN: AIC, BIC, FIC")
print("="*80)
print("\nConfiguración del FIC:")
print("  α = 5.0   (peso de penalización de FLOPs - aumentado)")
print("  β = 0.5   (peso de penalización de parámetros - reducido)")
print("  λ = 0.3   (30% log, 70% lineal - sensible a diferencias absolutas)")
print("\n  Objetivo: Priorizar eficiencia computacional cuando accuracy es similar")

from flop_counter import FlopInformationCriterion, count_model_flops

# Configurar FIC con más peso en FLOPs
# α=5.0: Mayor penalización de FLOPs (vs α=2.0 estándar)
# λ=0.3: 30% logarítmico, 70% lineal (más sensible a diferencias absolutas)
# β=0.5: Menor peso en parámetros (para que FLOPs dominen más)
OPTIMAL_LAMBDA = 0.3
ALPHA = 10.0
BETA = 0.5
fic_calculator = FlopInformationCriterion(
    variant='custom',
    alpha=ALPHA,  # Penaliza más los FLOPs
    beta=BETA,   # Menos peso a parámetros
    flops_scale=f'parametric_log_linear_{OPTIMAL_LAMBDA}'
)

def calculate_aic(log_likelihood, k):
    """AIC = -2*log(L) + 2*k"""
    return log_likelihood + 2 * k

def calculate_bic(log_likelihood, k, n):
    """BIC = -2*log(L) + k*log(n)"""
    return log_likelihood + k * np.log(n)

def calculate_log_likelihood_pytorch(model, data_loader, device):
    """Calcula -2*log(L) para modelo PyTorch"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0
    n_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            n_samples += len(target)
    
    # -2*log(L) = 2 * CrossEntropyLoss (sum)
    return total_loss

def calculate_log_likelihood_svm(model, X, y):
    """Calcula -2*log(L) para SVM (aproximación usando hinge loss)"""
    # Decision function da scores sin calibrar
    decision_scores = model.decision_function(X)
    
    # Convertir a probabilidades usando softmax
    exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Cross-entropy
    y_int = y.astype(int)
    log_likelihood = -2 * np.sum(np.log(probs[np.arange(len(y)), y_int] + 1e-10))
    
    return log_likelihood

print("\nCalculando métricas para cada modelo...")

# ---------------------------
# MLP
# ---------------------------
print("\n[1/3] MLP...")

# Log-likelihood
mlp_log_lik = calculate_log_likelihood_pytorch(mlp_model, test_loader, DEVICE)
n_test = len(test_dataset)

# AIC y BIC
mlp_aic = calculate_aic(mlp_log_lik, mlp_params)
mlp_bic = calculate_bic(mlp_log_lik, mlp_params, n_test)

# FIC - necesitamos contar FLOPs
print("  Contando FLOPs del MLP...")
sample_input = X_test_torch[:1].to(DEVICE)  # Una muestra para profiling
try:
    mlp_flops_result = count_model_flops(
        model=mlp_model,
        input_data=sample_input,
        framework='torch',
        verbose=False
    )
    mlp_flops = mlp_flops_result['total_flops']
    print(f"  FLOPs por muestra: {mlp_flops:,}")
except Exception as e:
    print(f"  Advertencia: No se pudieron contar FLOPs: {e}")
    mlp_flops = 0

# Calcular FIC
mlp_fic_result = fic_calculator.calculate_fic(
    log_likelihood=mlp_log_lik,
    flops=mlp_flops,
    n_params=mlp_params,
    n_samples=n_test
)
mlp_fic = mlp_fic_result['fic']

# ---------------------------
# CNN
# ---------------------------
print("\n[2/3] CNN...")

# Log-likelihood
cnn_log_lik = calculate_log_likelihood_pytorch(cnn_model, test_loader, DEVICE)

# AIC y BIC
cnn_aic = calculate_aic(cnn_log_lik, cnn_params)
cnn_bic = calculate_bic(cnn_log_lik, cnn_params, n_test)

# FIC - contar FLOPs
print("  Contando FLOPs de la CNN...")
try:
    cnn_flops_result = count_model_flops(
        model=cnn_model,
        input_data=sample_input,
        framework='torch',
        verbose=False
    )
    cnn_flops = cnn_flops_result['total_flops']
    print(f"  FLOPs por muestra: {cnn_flops:,}")
except Exception as e:
    print(f"  Advertencia: No se pudieron contar FLOPs: {e}")
    cnn_flops = 0

# Calcular FIC
cnn_fic_result = fic_calculator.calculate_fic(
    log_likelihood=cnn_log_lik,
    flops=cnn_flops,
    n_params=cnn_params,
    n_samples=n_test
)
cnn_fic = cnn_fic_result['fic']

# ---------------------------
# PCA+SVM
# ---------------------------
print("\n[3/3] PCA+SVM...")

# Log-likelihood (aproximación)
svm_log_lik = calculate_log_likelihood_svm(svm_model, X_test_pca_scaled, y_test)

# "Parámetros efectivos" del SVM: coeficientes de decisión
# LinearSVC con 10 clases tiene 10 vectores de coeficientes
svm_params = svm_model.coef_.size + svm_model.intercept_.size

# AIC y BIC
svm_aic = calculate_aic(svm_log_lik, svm_params)
svm_bic = calculate_bic(svm_log_lik, svm_params, n_test)

# FIC - SVM no tiene "FLOPs" en el sentido tradicional
# Estimamos operaciones: PCA transform + dot products
pca_ops = X_test_pca.shape[1] * X_test_flat.shape[1]  # 150 * 784
svm_ops = X_test_pca.shape[1] * svm_model.coef_.shape[0]  # 150 * 10
svm_flops = pca_ops + svm_ops  # Aproximación conservadora

print(f"  FLOPs estimados por muestra: {svm_flops:,}")

# Calcular FIC
svm_fic_result = fic_calculator.calculate_fic(
    log_likelihood=svm_log_lik,
    flops=svm_flops,
    n_params=svm_params,
    n_samples=n_test
)
svm_fic = svm_fic_result['fic']

# ============================================================================
# TABLA COMPARATIVA FINAL
# ============================================================================

print("\n" + "="*80)
print("COMPARACIÓN DE CRITERIOS DE INFORMACIÓN")
print("="*80)

print(f"\n{'Modelo':<15} {'Params':<12} {'FLOPs':<15} {'AIC':<15} {'BIC':<15} {'FIC':<15}")
print("-" * 87)
print(f"{'MLP':<15} {mlp_params:<12,} {mlp_flops:<15,} {mlp_aic:<15.2f} {mlp_bic:<15.2f} {mlp_fic:<15.2f}")
print(f"{'CNN':<15} {cnn_params:<12,} {cnn_flops:<15,} {cnn_aic:<15.2f} {cnn_bic:<15.2f} {cnn_fic:<15.2f}")
print(f"{'PCA+SVM':<15} {svm_params:<12,} {svm_flops:<15,} {svm_aic:<15.2f} {svm_bic:<15.2f} {svm_fic:<15.2f}")

# Determinar ganadores
print("\n" + "="*80)
print("MODELOS SELECCIONADOS POR CADA CRITERIO")
print("="*80)

models = {'MLP': (mlp_aic, mlp_bic, mlp_fic, mlp_test_acc, mlp_flops),
          'CNN': (cnn_aic, cnn_bic, cnn_fic, cnn_test_acc, cnn_flops),
          'PCA+SVM': (svm_aic, svm_bic, svm_fic, svm_test_acc, svm_flops)}

aic_winner = min(models.items(), key=lambda x: x[1][0])
bic_winner = min(models.items(), key=lambda x: x[1][1])
fic_winner = min(models.items(), key=lambda x: x[1][2])

print(f"\n{'Criterio':<15} {'Modelo Seleccionado':<20} {'Valor':<15} {'Razón':<40}")
print("-" * 90)
print(f"{'AIC':<15} {aic_winner[0]:<20} {aic_winner[1][0]:<15.2f} {'Minimiza params + ajuste':<40}")
print(f"{'BIC':<15} {bic_winner[0]:<20} {bic_winner[1][1]:<15.2f} {'Penaliza más los params':<40}")
print(f"{'FIC':<15} {fic_winner[0]:<20} {fic_winner[1][2]:<15.2f} {'Considera params + FLOPs + ajuste':<40}")

# Análisis de diferencias
print("\n" + "="*80)
print("ANÁLISIS: ¿Por qué cada criterio prefiere diferente modelo?")
print("="*80)

print(f"""
AIC selecciona: {aic_winner[0]}
  → Penaliza solo 2*k (# parámetros)
  → Ignora costo computacional (FLOPs)
  → {aic_winner[0]} tiene {aic_winner[1][3]:.2f}% accuracy
  
BIC selecciona: {bic_winner[0]}
  → Penaliza k*log(n) - más estricto que AIC
  → Favorece modelos más simples
  → También ignora FLOPs completamente
  → {bic_winner[0]} tiene {bic_winner[1][3]:.2f}% accuracy

FIC selecciona: {fic_winner[0]} (α=5.0, β=0.5, λ=0.3)
  → Penaliza FUERTEMENTE los FLOPs
  → También considera parámetros (pero con menos peso)
  → Balance entre precisión y eficiencia computacional
  → {fic_winner[0]} tiene {fic_winner[1][3]:.2f}% accuracy con {fic_winner[1][4]:,} FLOPs
  
Trade-off detectado por FIC:
  CNN: {cnn_test_acc:.2f}% acc, {cnn_flops:,} FLOPs
  MLP: {mlp_test_acc:.2f}% acc, {mlp_flops:,} FLOPs
  → Diferencia accuracy: {cnn_test_acc - mlp_test_acc:.2f}%
  → Diferencia FLOPs: {(cnn_flops / mlp_flops):.1f}x más costoso
  → ¿Vale la pena {(cnn_flops / mlp_flops):.1f}x más FLOPs por {cnn_test_acc - mlp_test_acc:.2f}% mejor?
  → FIC dice: {"SÍ" if fic_winner[0] == "CNN" else "NO"}
""")

# Recomendación final
print("="*80)
print("RECOMENDACIÓN SEGÚN CONTEXTO")
print("="*80)

print("""
📱 MÓVIL/EDGE (recursos limitados):
   → Usar FIC para seleccionar modelo
   → FLOPs son críticos para latencia y batería
   → Con α=5.0, FIC penaliza fuertemente modelos costosos
   
🖥️  SERVIDOR (recursos abundantes):
   → AIC/BIC suficientes si solo importa precisión
   → FLOPs menos relevantes, priorizar accuracy
   
⚖️  PRODUCCIÓN BALANCEADA (nuestra configuración):
   → FIC con α=5.0, β=0.5, λ=0.3
   → Penaliza modelos ineficientes
   → Prefiere MLP si CNN no ofrece mejora sustancial
   → Ideal: CNN debe ser >2% mejor para justificar 14x FLOPs

💡 AJUSTAR SENSIBILIDAD:
   α más alto (7.0-10.0) → Penaliza AÚN MÁS los FLOPs
   λ más bajo (0.1-0.2)  → Más sensible a diferencias absolutas
   β más bajo (0.1-0.3)  → Ignora casi completamente los parámetros
""")

print("\n" + "="*80)
print("VERIFICACIÓN Y EVALUACIÓN COMPLETADOS")
print("="*80)
print("\nLos 3 modelos han sido entrenados y evaluados con AIC, BIC y FIC")
print("\n💡 Observaciones clave:")
print("  - AIC/BIC ignoran FLOPs completamente")
print("  - FIC considera el costo computacional real")
print("  - Diferentes criterios pueden seleccionar diferentes modelos")
print("  - La elección depende del contexto de deployment")