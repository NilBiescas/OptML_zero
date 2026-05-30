"""CPU smoke tests for the three new zeroth-order optimizers.

Runs 3 steps of each on a tiny two-linear-layer model and verifies:
    - closure executes without error
    - parameters change after step()
    - loss values are finite
    - state buffers exist with the expected shapes
    - reproducibility: same seed -> same final parameters

No GPU required, no real dataset. This file is run as part of the static
verification suite (see plan §8) and is not used in training.
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.conmezo import ConMeZO
from optimizers.fzoo import FZOO
from optimizers.zo_muon import ZOMuon


def _make_model(seed: int = 0, device: str = "cpu",
                dtype: torch.dtype = torch.float32) -> nn.Module:
    """Tiny model: 8 -> 16 -> 4, two Linear layers (each has a 2D weight and
    a 1D bias) so we exercise both the 2D and 1D code paths.
    """
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    ).to(device=device, dtype=dtype)
    # train.py injects param_id on each requires_grad parameter; mirror that.
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            p.param_id = i
    return model


def _make_closure(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """A closure mirroring train.py's: returns a scalar loss tensor under
    no_grad. Uses MSE because ZO doesn't care about the loss family.
    """
    def closure():
        with torch.no_grad():
            out = model(x)
            return torch.nn.functional.mse_loss(out, y)
    return closure


def _params_snapshot(model: nn.Module):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _params_changed(before, after):
    return any(not torch.allclose(b, a) for b, a in zip(before, after))


def _run_optimizer(opt_factory, n_steps: int = 3):
    model = _make_model(seed=0)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    closure = _make_closure(model, x, y)
    opt = opt_factory(model)

    losses = []
    before = _params_snapshot(model)
    for _ in range(n_steps):
        loss = opt.step(closure)
        losses.append(loss)
    after = _params_snapshot(model)

    return model, opt, losses, before, after


# -----------------------------------------------------------------------
# ConMeZO
# -----------------------------------------------------------------------
def test_conmezo_runs():
    factory = lambda m: ConMeZO(m.parameters(), lr=1e-4, eps=1e-3,
                                cone_theta=1.4, cone_beta=0.99, seed=42)
    model, opt, losses, before, after = _run_optimizer(factory)
    assert all(math.isfinite(L) for L in losses), f"non-finite loss: {losses}"
    assert _params_changed(before, after), "ConMeZO did not change params"
    # State checks: every param should have an 'mu' buffer of matching shape.
    for p in model.parameters():
        if not p.requires_grad:
            continue
        st = opt.state[p]
        assert 'mu' in st, "missing mu buffer"
        assert st['mu'].shape == p.shape, f"mu shape mismatch: {st['mu'].shape} vs {p.shape}"
        assert st['step'] == 3, f"step should be 3, got {st['step']}"


def test_conmezo_reproducible():
    factory = lambda m: ConMeZO(m.parameters(), lr=1e-4, eps=1e-3,
                                cone_theta=1.4, cone_beta=0.99, seed=42)
    _, _, _, _, after1 = _run_optimizer(factory)
    _, _, _, _, after2 = _run_optimizer(factory)
    for a, b in zip(after1, after2):
        assert torch.allclose(a, b), "ConMeZO not reproducible across runs"


# -----------------------------------------------------------------------
# FZOO
# -----------------------------------------------------------------------
def test_fzoo_runs():
    factory = lambda m: FZOO(m.parameters(), lr=1e-4, eps=1e-3,
                             Nq=4, sigma_floor=1e-6, seed=42)
    model, opt, losses, before, after = _run_optimizer(factory)
    assert all(math.isfinite(L) for L in losses), f"non-finite loss: {losses}"
    assert _params_changed(before, after), "FZOO did not change params"
    for p in model.parameters():
        if not p.requires_grad:
            continue
        st = opt.state[p]
        # 'acc' is transient: cleared at end of step.
        assert 'acc' not in st, "acc buffer should be cleared after step"
        assert st['step'] == 3, f"step should be 3, got {st['step']}"


def test_fzoo_Nq1():
    # Edge case: Nq=1 means sigma_hat falls back to |c_0|.
    factory = lambda m: FZOO(m.parameters(), lr=1e-4, eps=1e-3,
                             Nq=1, sigma_floor=1e-6, seed=42)
    _, _, losses, before, after = _run_optimizer(factory)
    assert all(math.isfinite(L) for L in losses)
    assert _params_changed(before, after)


def test_fzoo_reproducible():
    factory = lambda m: FZOO(m.parameters(), lr=1e-4, eps=1e-3,
                             Nq=4, sigma_floor=1e-6, seed=42)
    _, _, _, _, after1 = _run_optimizer(factory)
    _, _, _, _, after2 = _run_optimizer(factory)
    for a, b in zip(after1, after2):
        assert torch.allclose(a, b), "FZOO not reproducible across runs"


# -----------------------------------------------------------------------
# ZOMuon
# -----------------------------------------------------------------------
def test_zomuon_runs():
    factory = lambda m: ZOMuon(m.parameters(), lr=1e-3, eps=1e-3,
                               r=4, Nq=2, ns_steps=5, refresh_T=2,
                               momentum=0.9, seed=42)
    model, opt, losses, before, after = _run_optimizer(factory)
    assert all(math.isfinite(L) for L in losses), f"non-finite loss: {losses}"
    assert _params_changed(before, after), "ZOMuon did not change params"
    for p in model.parameters():
        if not p.requires_grad:
            continue
        st = opt.state[p]
        if p.dim() >= 2:
            assert 'P' in st, "missing P buffer for 2D weight"
            assert st['P'].shape[1] == 4, f"P rank mismatch: {st['P'].shape}"
            assert 'momentum_buf' in st, "missing momentum_buf for 2D weight"
            assert st['momentum_buf'].shape == p.shape
        assert st['step'] == 3, f"step should be 3, got {st['step']}"


def test_zomuon_reproducible():
    factory = lambda m: ZOMuon(m.parameters(), lr=1e-3, eps=1e-3,
                               r=4, Nq=2, ns_steps=5, refresh_T=2,
                               momentum=0.9, seed=42)
    _, _, _, _, after1 = _run_optimizer(factory)
    _, _, _, _, after2 = _run_optimizer(factory)
    for a, b in zip(after1, after2):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), "ZOMuon not reproducible"


def test_zomuon_P_refresh():
    """P should be sampled at step 0 and re-sampled when step % refresh_T == 0."""
    factory = lambda m: ZOMuon(m.parameters(), lr=1e-3, eps=1e-3,
                               r=4, Nq=2, ns_steps=5, refresh_T=2,
                               momentum=0.9, seed=42)
    model = _make_model(seed=0)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    closure = _make_closure(model, x, y)
    opt = factory(model)

    # Step 0: P initialised.
    opt.step(closure)
    P0 = {id(p): opt.state[p]['P'].clone()
          for p in model.parameters()
          if p.requires_grad and p.dim() >= 2}
    # Step 1: refresh_T=2, step%2 != 0 -> P stays the same.
    opt.step(closure)
    for p in model.parameters():
        if p.requires_grad and p.dim() >= 2:
            assert torch.allclose(P0[id(p)], opt.state[p]['P']), \
                "P should not refresh at step 1 with refresh_T=2"
    # Step 2: step%2 == 0 -> P refreshes.
    opt.step(closure)
    any_changed = False
    for p in model.parameters():
        if p.requires_grad and p.dim() >= 2:
            if not torch.allclose(P0[id(p)], opt.state[p]['P']):
                any_changed = True
    assert any_changed, "P should refresh at step 2 with refresh_T=2"


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
def main():
    tests = [
        ("ConMeZO runs",         test_conmezo_runs),
        ("ConMeZO reproducible", test_conmezo_reproducible),
        ("FZOO runs",            test_fzoo_runs),
        ("FZOO Nq=1",            test_fzoo_Nq1),
        ("FZOO reproducible",    test_fzoo_reproducible),
        ("ZOMuon runs",          test_zomuon_runs),
        ("ZOMuon reproducible",  test_zomuon_reproducible),
        ("ZOMuon P refresh",     test_zomuon_P_refresh),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
