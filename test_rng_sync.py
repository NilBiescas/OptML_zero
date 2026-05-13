import torch
gen0 = torch.Generator(device='cuda:0').manual_seed(42)
gen1 = torch.Generator(device='cuda:0').manual_seed(42)
t0 = torch.randn(5, generator=gen0)
t1 = torch.randn(5, generator=gen1)
print(f"CUDA:0 vs CUDA:0 -> {torch.allclose(t0, t1)}")

if torch.cuda.device_count() > 1:
    gen2 = torch.Generator(device='cuda:1').manual_seed(42)
    t2 = torch.randn(5, device='cuda:1', generator=gen2)
    print(f"CUDA:0 vs CUDA:1 -> {torch.allclose(t0, t2.to('cuda:0'))}")
