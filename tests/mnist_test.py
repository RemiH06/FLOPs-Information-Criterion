#!/usr/bin/env python3
"""
MNIST Classification: Comparación exhaustiva de 10 enfoques diferentes
Objetivo: Demostrar la robustez del FIC en un problema real con múltiples arquitecturas
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time

print("="*100)
print("MNIST: COMPARACIÓN EXHAUSTIVA DE 10 ENFOQUES")
print("="*100)

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

print("\n" + "="*100)
print("CARGANDO DATOS MNIST")
print("="*100)

print("Descargando MNIST desde OpenML...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')

X = mnist.data.to_numpy().astype(np.float32)
y = mnist.target.to_numpy().astype(np.int64)

# Normalizar
X = X / 255.0

# Split train/test
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
# FUNCIONES AUXILIARES
# ============================================================================

def count_parameters(model):
    """Cuenta parámetros entrenables"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_pytorch_model(model, train_loader, epochs, lr):
    """Entrena un modelo de PyTorch"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_pytorch_model(model, test_loader):
    """Evalúa un modelo de PyTorch"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total

# ============================================================================
# MODELO 1: Regresión Logística (Baseline clásico)
# ============================================================================

print("\n" + "="*100)
print("MODELO 1: REGRESIÓN LOGÍSTICA")
print("="*100)
print("Características: Lineal, sin capas ocultas, baseline clásico")

logreg_start = time.time()

logreg_model = LogisticRegression(max_iter=100, random_state=42, verbose=0)
logreg_model.fit(X_train, y_train)

logreg_train_time = time.time() - logreg_start
logreg_predictions = logreg_model.predict(X_test)
logreg_accuracy = 100. * np.mean(logreg_predictions == y_test)

logreg_params = logreg_model.coef_.size + logreg_model.intercept_.size

print(f"\nResultados:")
print(f"  Test Accuracy: {logreg_accuracy:.2f}%")
print(f"  Parámetros: {logreg_params:,}")
print(f"  Training Time: {logreg_train_time:.1f}s")

# ============================================================================
# MODELO 2: K-Nearest Neighbors
# ============================================================================

print("\n" + "="*100)
print("MODELO 2: K-NEAREST NEIGHBORS (k=5)")
print("="*100)
print("Características: No paramétrico, basado en distancia")

knn_start = time.time()

# Usar subset para acelerar
subset_size = 10000
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train[:subset_size], y_train[:subset_size])

knn_train_time = time.time() - knn_start
knn_predictions = knn_model.predict(X_test)
knn_accuracy = 100. * np.mean(knn_predictions == y_test)

knn_params = 0  # KNN no tiene parámetros tradicionales

print(f"\nResultados:")
print(f"  Test Accuracy: {knn_accuracy:.2f}%")
print(f"  Parámetros: {knn_params} (no paramétrico)")
print(f"  Training Time: {knn_train_time:.1f}s")
print(f"  Nota: Entrenado con subset de {subset_size} muestras")

# ============================================================================
# MODELO 3: Random Forest
# ============================================================================

print("\n" + "="*100)
print("MODELO 3: RANDOM FOREST (100 árboles)")
print("="*100)
print("Características: Ensemble, robusto, interpretable")

rf_start = time.time()

rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_train_time = time.time() - rf_start
rf_predictions = rf_model.predict(X_test)
rf_accuracy = 100. * np.mean(rf_predictions == y_test)

# Estimación de parámetros: nodos × árboles
rf_params = sum(tree.tree_.node_count for tree in rf_model.estimators_)

print(f"\nResultados:")
print(f"  Test Accuracy: {rf_accuracy:.2f}%")
print(f"  Parámetros: {rf_params:,} (nodos totales)")
print(f"  Training Time: {rf_train_time:.1f}s")

# ============================================================================
# MODELO 4: PCA + SVM
# ============================================================================

print("\n" + "="*100)
print("MODELO 4: PCA (150 componentes) + LINEAR SVM")
print("="*100)
print("Características: Reducción dimensionalidad + clasificador lineal")

pca_svm_start = time.time()

# PCA
pca = PCA(n_components=150, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# SVM
scaler = StandardScaler()
X_train_pca_scaled = scaler.fit_transform(X_train_pca)
X_test_pca_scaled = scaler.transform(X_test_pca)

svm_model = LinearSVC(C=1.0, max_iter=1000, random_state=42)
svm_model.fit(X_train_pca_scaled, y_train)

pca_svm_train_time = time.time() - pca_svm_start
pca_svm_predictions = svm_model.predict(X_test_pca_scaled)
pca_svm_accuracy = 100. * np.mean(pca_svm_predictions == y_test)

pca_svm_params = svm_model.coef_.size + svm_model.intercept_.size

print(f"\nResultados:")
print(f"  Test Accuracy: {pca_svm_accuracy:.2f}%")
print(f"  Parámetros: {pca_svm_params:,}")
print(f"  Training Time: {pca_svm_train_time:.1f}s")
print(f"  Varianza explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ============================================================================
# MODELO 5: MLP Tiny (muy pequeño)
# ============================================================================

print("\n" + "="*100)
print("MODELO 5: MLP TINY (784 -> 64 -> 10)")
print("="*100)
print("Características: Red mínima, ultra eficiente")

class MLPTiny(nn.Module):
    def __init__(self):
        super(MLPTiny, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

mlp_tiny_model = MLPTiny().to(DEVICE)
mlp_tiny_params = count_parameters(mlp_tiny_model)

print(f"\nParámetros: {mlp_tiny_params:,}")
print("Entrenando...")

mlp_tiny_start = time.time()
train_pytorch_model(mlp_tiny_model, train_loader, EPOCHS, LEARNING_RATE)
mlp_tiny_train_time = time.time() - mlp_tiny_start

mlp_tiny_accuracy = evaluate_pytorch_model(mlp_tiny_model, test_loader)

print(f"\nResultados:")
print(f"  Test Accuracy: {mlp_tiny_accuracy:.2f}%")
print(f"  Training Time: {mlp_tiny_train_time:.1f}s")

# ============================================================================
# MODELO 6: MLP Medium (balanceado)
# ============================================================================

print("\n" + "="*100)
print("MODELO 6: MLP MEDIUM (784 -> 256 -> 128 -> 10)")
print("="*100)
print("Características: Balance entre tamaño y capacidad")

class MLPMedium(nn.Module):
    def __init__(self):
        super(MLPMedium, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

mlp_medium_model = MLPMedium().to(DEVICE)
mlp_medium_params = count_parameters(mlp_medium_model)

print(f"\nParámetros: {mlp_medium_params:,}")
print("Entrenando...")

mlp_medium_start = time.time()
train_pytorch_model(mlp_medium_model, train_loader, EPOCHS, LEARNING_RATE)
mlp_medium_train_time = time.time() - mlp_medium_start

mlp_medium_accuracy = evaluate_pytorch_model(mlp_medium_model, test_loader)

print(f"\nResultados:")
print(f"  Test Accuracy: {mlp_medium_accuracy:.2f}%")
print(f"  Training Time: {mlp_medium_train_time:.1f}s")

# ============================================================================
# MODELO 7: MLP Large (sobredimensionado)
# ============================================================================

print("\n" + "="*100)
print("MODELO 7: MLP LARGE (784 -> 512 -> 512 -> 256 -> 10)")
print("="*100)
print("Características: Muy grande, posible overfitting")

class MLPLarge(nn.Module):
    def __init__(self):
        super(MLPLarge, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

mlp_large_model = MLPLarge().to(DEVICE)
mlp_large_params = count_parameters(mlp_large_model)

print(f"\nParámetros: {mlp_large_params:,}")
print("Entrenando...")

mlp_large_start = time.time()
train_pytorch_model(mlp_large_model, train_loader, EPOCHS, LEARNING_RATE)
mlp_large_train_time = time.time() - mlp_large_start

mlp_large_accuracy = evaluate_pytorch_model(mlp_large_model, test_loader)

print(f"\nResultados:")
print(f"  Test Accuracy: {mlp_large_accuracy:.2f}%")
print(f"  Training Time: {mlp_large_train_time:.1f}s")

# ============================================================================
# MODELO 8: CNN Tiny (mínima convolucional)
# ============================================================================

print("\n" + "="*100)
print("MODELO 8: CNN TINY (1 capa conv)")
print("="*100)
print("Características: Convolucional mínima, muy eficiente")

class CNNTiny(nn.Module):
    def __init__(self):
        super(CNNTiny, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

cnn_tiny_model = CNNTiny().to(DEVICE)
cnn_tiny_params = count_parameters(cnn_tiny_model)

print(f"\nParámetros: {cnn_tiny_params:,}")
print("Entrenando...")

cnn_tiny_start = time.time()
train_pytorch_model(cnn_tiny_model, train_loader, EPOCHS, LEARNING_RATE)
cnn_tiny_train_time = time.time() - cnn_tiny_start

cnn_tiny_accuracy = evaluate_pytorch_model(cnn_tiny_model, test_loader)

print(f"\nResultados:")
print(f"  Test Accuracy: {cnn_tiny_accuracy:.2f}%")
print(f"  Training Time: {cnn_tiny_train_time:.1f}s")

# ============================================================================
# MODELO 9: CNN Medium (LeNet-like)
# ============================================================================

print("\n" + "="*100)
print("MODELO 9: CNN MEDIUM (2 capas conv con BatchNorm)")
print("="*100)
print("Características: LeNet-like moderno, buen balance")

class CNNMedium(nn.Module):
    def __init__(self):
        super(CNNMedium, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

cnn_medium_model = CNNMedium().to(DEVICE)
cnn_medium_params = count_parameters(cnn_medium_model)

print(f"\nParámetros: {cnn_medium_params:,}")
print("Entrenando...")

cnn_medium_start = time.time()
train_pytorch_model(cnn_medium_model, train_loader, EPOCHS, LEARNING_RATE)
cnn_medium_train_time = time.time() - cnn_medium_start

cnn_medium_accuracy = evaluate_pytorch_model(cnn_medium_model, test_loader)

print(f"\nResultados:")
print(f"  Test Accuracy: {cnn_medium_accuracy:.2f}%")
print(f"  Training Time: {cnn_medium_train_time:.1f}s")

# ============================================================================
# MODELO 10: CNN Deep (muy profunda)
# ============================================================================

print("\n" + "="*100)
print("MODELO 10: CNN DEEP (4 capas conv, muy profunda)")
print("="*100)
print("Características: Máxima capacidad, posible overkill para MNIST")

class CNNDeep(nn.Module):
    def __init__(self):
        super(CNNDeep, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 1 * 1, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

cnn_deep_model = CNNDeep().to(DEVICE)
cnn_deep_params = count_parameters(cnn_deep_model)

print(f"\nParámetros: {cnn_deep_params:,}")
print("Entrenando...")

cnn_deep_start = time.time()
train_pytorch_model(cnn_deep_model, train_loader, EPOCHS, LEARNING_RATE)
cnn_deep_train_time = time.time() - cnn_deep_start

cnn_deep_accuracy = evaluate_pytorch_model(cnn_deep_model, test_loader)

print(f"\nResultados:")
print(f"  Test Accuracy: {cnn_deep_accuracy:.2f}%")
print(f"  Training Time: {cnn_deep_train_time:.1f}s")

# ============================================================================
# CALCULAR AIC, BIC Y FIC PARA TODOS LOS MODELOS
# ============================================================================

print("\n" + "="*100)
print("CALCULANDO CRITERIOS DE INFORMACIÓN: AIC, BIC, FIC")
print("="*100)

from flop_counter import FlopInformationCriterion, count_model_flops

# ============================================================================
# CONFIGURACIÓN FIC - AJUSTAR ESTOS PARÁMETROS
# ============================================================================

print("\n" + "-"*100)
print("CONFIGURACIÓN DE PARÁMETROS FIC")
print("-"*100)

# Parámetros ajustables del FIC
FIC_VARIANT = 'custom'  # Opciones: 'deployment', 'research', 'balanced', 'custom'
FIC_ALPHA = 5.0         # Peso del término de FLOPs (mayor = penaliza más FLOPs)
FIC_BETA = 0.5          # Peso del término de parámetros (mayor = penaliza más params)
FIC_LAMBDA = 0.3        # Balance entre términos (0=solo FLOPs, 1=solo params)

print(f"\nParámetros FIC configurados:")
print(f"  Variant:        {FIC_VARIANT}")
print(f"  Alpha (FLOPs):  {FIC_ALPHA}")
print(f"  Beta (Params):  {FIC_BETA}")
print(f"  Lambda (Balance): {FIC_LAMBDA}")

print("\n💡 Guía de ajuste:")
print("  • ALPHA alto → Penaliza fuertemente modelos con muchos FLOPs")
print("  • BETA alto → Penaliza fuertemente modelos con muchos parámetros")
print("  • LAMBDA cercano a 0 → Prioriza eficiencia en FLOPs")
print("  • LAMBDA cercano a 1 → Prioriza parsimonia en parámetros")
print("\nEjemplos de configuraciones:")
print("  Deployment (edge):   alpha=10.0, beta=1.0, lambda=0.2")
print("  Research:            alpha=1.0, beta=2.0, lambda=0.7")
print("  Balanced:            alpha=5.0, beta=0.5, lambda=0.3")

# Configurar FIC con parámetros personalizados
fic_calculator = FlopInformationCriterion(
    variant=FIC_VARIANT,
    alpha=FIC_ALPHA,
    beta=FIC_BETA,
    lambda_balance=FIC_LAMBDA
)

# ============================================================================
# FUNCIONES DE CÁLCULO
# ============================================================================

def calculate_aic(log_likelihood, k):
    """AIC = 2k - 2ln(L) = log_likelihood + 2k"""
    return log_likelihood + 2 * k

def calculate_bic(log_likelihood, k, n):
    """BIC = k*ln(n) - 2ln(L) = log_likelihood + k*ln(n)"""
    return log_likelihood + k * np.log(n)

def calculate_log_likelihood_pytorch(model, data_loader, device):
    """Calcula log-likelihood negativa para modelos PyTorch"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
    
    return total_loss

def calculate_log_likelihood_sklearn(predictions, y_true, n_classes=10):
    """Aproximación de log-likelihood para modelos sklearn"""
    correct = (predictions == y_true).astype(float)
    # Probabilidades: alta si correcto, baja si incorrecto
    probs = np.where(correct == 1, 0.99, 0.01/(n_classes-1))
    return -2 * np.sum(np.log(probs + 1e-10))

# ============================================================================
# EVALUAR MODELOS SKLEARN
# ============================================================================

all_results = {}

print("\n" + "="*100)
print("EVALUANDO MODELOS SKLEARN")
print("="*100)

sklearn_models = [
    ("Logistic Regression", logreg_predictions, logreg_params, logreg_accuracy, logreg_train_time),
    ("KNN (k=5)", knn_predictions, knn_params if knn_params > 0 else 100, knn_accuracy, knn_train_time),
    ("Random Forest", rf_predictions, rf_params, rf_accuracy, rf_train_time),
    ("PCA+SVM", pca_svm_predictions, pca_svm_params, pca_svm_accuracy, pca_svm_train_time),
]

for name, predictions, n_params, accuracy, train_time in sklearn_models:
    print(f"\n[sklearn] {name}...")
    
    # Calcular log-likelihood
    log_lik = calculate_log_likelihood_sklearn(predictions, y_test)
    
    # Calcular AIC y BIC
    aic = calculate_aic(log_lik, n_params)
    bic = calculate_bic(log_lik, n_params, len(y_test))
    
    # Estimar FLOPs según el tipo de modelo
    if "Logistic" in name:
        flops = 784 * 10 * 2  # matmul input->output
    elif "KNN" in name:
        flops = 784 * len(y_test) * 5  # distance calculations
    elif "Forest" in name:
        flops = rf_params * 10  # traversals aproximados
    elif "SVM" in name:
        flops = 150 * 10 * 2 + 784 * 150  # PCA transform + SVM
    else:
        flops = 0
    
    # Calcular FIC con parámetros configurados
    fic_result = fic_calculator.calculate_fic(
        log_likelihood=log_lik,
        flops=flops,
        n_params=n_params,
        n_samples=len(y_test)
    )
    
    # Mostrar desglose del FIC
    print(f"  Log-likelihood: {log_lik:,.0f}")
    print(f"  Params: {n_params:,}")
    print(f"  FLOPs: {flops:,}")
    print(f"  AIC: {aic:,.0f}")
    print(f"  BIC: {bic:,.0f}")
    print(f"  FIC: {fic_result['fic']:,.0f}")
    print(f"    └─ FLOPs term: {fic_result.get('flops_term', 0):,.0f}")
    print(f"    └─ Params term: {fic_result.get('params_term', 0):,.0f}")
    
    all_results[name] = {
        'accuracy': accuracy,
        'params': n_params,
        'flops': flops,
        'train_time': train_time,
        'aic': aic,
        'bic': bic,
        'fic': fic_result['fic'],
        'log_likelihood': log_lik,
        'flops_term': fic_result.get('flops_term', 0),
        'params_term': fic_result.get('params_term', 0)
    }

# ============================================================================
# EVALUAR MODELOS PYTORCH
# ============================================================================

print("\n" + "="*100)
print("EVALUANDO MODELOS PYTORCH")
print("="*100)

pytorch_models = [
    ("MLP Tiny", mlp_tiny_model, mlp_tiny_params, mlp_tiny_accuracy, mlp_tiny_train_time),
    ("MLP Medium", mlp_medium_model, mlp_medium_params, mlp_medium_accuracy, mlp_medium_train_time),
    ("MLP Large", mlp_large_model, mlp_large_params, mlp_large_accuracy, mlp_large_train_time),
    ("CNN Tiny", cnn_tiny_model, cnn_tiny_params, cnn_tiny_accuracy, cnn_tiny_train_time),
    ("CNN Medium", cnn_medium_model, cnn_medium_params, cnn_medium_accuracy, cnn_medium_train_time),
    ("CNN Deep", cnn_deep_model, cnn_deep_params, cnn_deep_accuracy, cnn_deep_train_time),
]

for name, model, n_params, accuracy, train_time in pytorch_models:
    print(f"\n[PyTorch] {name}...")
    
    # Calcular log-likelihood
    log_lik = calculate_log_likelihood_pytorch(model, test_loader, DEVICE)
    
    # Calcular AIC y BIC
    aic = calculate_aic(log_lik, n_params)
    bic = calculate_bic(log_lik, n_params, len(y_test))
    
    # Contar FLOPs reales
    sample_input = X_test_torch[:1].to(DEVICE)
    try:
        flops_result = count_model_flops(
            model=model,
            input_data=sample_input,
            framework='torch',
            verbose=False
        )
        flops = flops_result['total_flops']
    except Exception as e:
        print(f"  ⚠️  Warning: No se pudieron contar FLOPs: {e}")
        flops = 0
    
    # Calcular FIC con parámetros configurados
    fic_result = fic_calculator.calculate_fic(
        log_likelihood=log_lik,
        flops=flops,
        n_params=n_params,
        n_samples=len(y_test)
    )
    
    # Mostrar desglose del FIC
    print(f"  Log-likelihood: {log_lik:,.0f}")
    print(f"  Params: {n_params:,}")
    print(f"  FLOPs: {flops:,}")
    print(f"  AIC: {aic:,.0f}")
    print(f"  BIC: {bic:,.0f}")
    print(f"  FIC: {fic_result['fic']:,.0f}")
    print(f"    └─ FLOPs term: {fic_result.get('flops_term', 0):,.0f}")
    print(f"    └─ Params term: {fic_result.get('params_term', 0):,.0f}")
    
    all_results[name] = {
        'accuracy': accuracy,
        'params': n_params,
        'flops': flops,
        'train_time': train_time,
        'aic': aic,
        'bic': bic,
        'fic': fic_result['fic'],
        'log_likelihood': log_lik,
        'flops_term': fic_result.get('flops_term', 0),
        'params_term': fic_result.get('params_term', 0)
    }

# ============================================================================
# TABLA COMPARATIVA FINAL
# ============================================================================

print("\n" + "="*100)
print("TABLA COMPARATIVA COMPLETA: 10 ENFOQUES")
print("="*100)

print(f"\n{'Modelo':<25} {'Acc%':<8} {'Params':<12} {'FLOPs':<15} {'Time(s)':<10} {'AIC':<12} {'BIC':<12} {'FIC':<12}")
print("-" * 120)

for name in sorted(all_results.keys(), key=lambda x: all_results[x]['fic']):
    r = all_results[name]
    print(f"{name:<25} {r['accuracy']:<8.2f} {r['params']:<12,} {r['flops']:<15,} "
          f"{r['train_time']:<10.1f} {r['aic']:<12.0f} {r['bic']:<12.0f} {r['fic']:<12.0f}")