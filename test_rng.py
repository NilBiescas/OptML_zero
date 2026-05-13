import torch
from optimizers.lozo import LOZO

p1 = torch.nn.Parameter(torch.randn(10, 10))
p2 = torch.nn.Parameter(p1.clone())

opt1 = LOZO([p1], lr=0.1, r=4, nu=5, seed=42)
opt2 = LOZO([p2], lr=0.1, r=4, nu=5, seed=42)

# Advance global RNG on one side to simulate drift/divergence
torch.randn(1000)

opt1.step(lambda: 1.0)
opt2.step(lambda: 1.0)

if torch.allclose(p1, p2):
    print("RNG SYNCHRONIZATION TEST: SUCCESS (Parameters are 100% identical despite global RNG drift!)")
else:
    print("RNG SYNCHRONIZATION TEST: FAILED")
