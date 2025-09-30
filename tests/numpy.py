import flop_counter
import numpy as np

# Definir modelo
def my_model(x):
    W1 = np.random.randn(100, 50)
    W2 = np.random.randn(50, 10)
    h = np.matmul(x, W1)
    return np.matmul(h, W2)

# Datos de entrada
x = np.random.randn(32, 100)

# Contar FLOPs con una sola línea
result = flop_counter.count_model_flops(my_model, x)

print(f"FLOPs: {result['total_flops']:,}")