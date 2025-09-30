import torch
import torch.nn as nn
import flop_counter

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(32, 784)

# Detección automática de framework
result = flop_counter.count_model_flops(model, x)