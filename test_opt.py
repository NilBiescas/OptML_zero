import torch
from optimizers.lozo import LOZO, LOZOM

model = torch.nn.Linear(10, 10)
# Add a bias to test 1D parameters too
bias = torch.nn.Parameter(torch.zeros(10))
model.register_parameter('bias_test', bias)

opt1 = LOZO(model.parameters(), lr=0.1, r=4, nu=5, seed=42)
opt2 = LOZOM(model.parameters(), lr=0.1, r=4, nu=5, beta=0.9, seed=42)

def closure():
    # compute some dummy output loss
    x = torch.randn(1, 10)
    out = model(x).sum() + model.bias_test.sum()
    return out

try:
    opt1.step(closure)
    print("LOZO step: SUCCESS")
    opt2.step(closure)
    print("LOZOM step: SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
