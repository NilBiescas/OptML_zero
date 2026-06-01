# New Methods — OptML_zero

Plan for the next phase of the project: **reproduce three published, working zeroth-order (ZO) optimization methods** that are not yet in the repo, then study combinations.

Scope: *no novel-port speculation*. Every method listed below is a published paper with reported gains over MeZO, and (where possible) public code.

---

## 1. What is already implemented (oldest → newest)

All six live under [`optimizers/`](optimizers/) and are wired into `train.py` via `config*.yaml`.

| # | Optimizer | File | Year / Venue | Paper |
|---|-----------|------|--------------|-------|
| 1 | **MeZO** — baseline SPSA, full-rank `±ε z` | `optimizers/mezo.py` | NeurIPS **2023** | Malladi et al. — [arXiv:2305.17333](https://arxiv.org/abs/2305.17333) |
| 2 | **Sparse-MeZO** — masked perturbation (magnitude / random) | `optimizers/sparse_mezo.py` | Feb **2024** | Liu et al. — [arXiv:2402.15751](https://arxiv.org/abs/2402.15751) |
| 3 | **HiZOO** — diagonal-Hessian-preconditioned SPSA | `optimizers/hizoo.py` | NeurIPS **2024** | Zhao et al. — [arXiv:2402.15173](https://arxiv.org/abs/2402.15173) |
| 4 | **LOZO** — low-rank subspace `ε U Vᵀ`, lazy resampling of V | `optimizers/lozo.py` | ICLR **2025** | Chen et al. — [arXiv:2410.07698](https://arxiv.org/abs/2410.07698) |
| 5 | **LOZO-M** — LOZO + subspace-projected momentum | `optimizers/lozo.py` | ICLR **2025** | Same paper — [arXiv:2410.07698](https://arxiv.org/abs/2410.07698) |
| 6 | **DiZO** — divergence-rescaled, per-layer α projection | `optimizers/dizo.py` | NeurIPS **2025** | *Harmony in Divergence* — [arXiv:2502.03304](https://arxiv.org/abs/2502.03304) |

### Author attribution (from `git log` on `optimizers/`)

| Author | Files |
|--------|-------|
| NilBiescas / Nil | `lozo.py` (LOZO, LOZO-M + many multi-GPU RNG and momentum fixes) |
| mpilligua | `sparse_mezo.py` |
| **ChenghengLi (you)** | `mezo.py`, `hizoo.py`, `dizo.py` |

Of your three, the only standalone novel-algorithm contribution is **DiZO**; the other two are baseline ports.

---

## 2. Top 3 methods to reproduce next  —  *picked for VARIETY*

> **Selection rule.** We already have MeZO, Sparse-MeZO, HiZOO, LOZO, LOZO-M, DiZO. So we cover: full-rank baseline / sparse / diagonal-Hessian / low-rank subspace / momentum / per-layer rescaling. The next three methods must each occupy an **empty axis** — implementing yet another "HiZOO variant" or "LOZO variant" adds little. Below, each pick is paired with the axis it opens.

### Coverage map
| Axis | Already in repo? | Methods occupying it |
|------|------------------|---------------------|
| Full-rank SPSA baseline | ✅ | MeZO |
| Sparse / masked perturbation | ✅ | Sparse-MeZO |
| Diagonal-Hessian preconditioning | ✅ | HiZOO  (HELENE / LOREN would only refine this) |
| Random low-rank subspace | ✅ | LOZO  (SubZero / P-GAP only refine this) |
| Low-rank + momentum | ✅ | LOZO-M |
| Per-layer divergence rescaling | ✅ | DiZO |
| **2D matrix structure / spectral** | ❌ | — |
| **Perturbation distribution shape** | ❌ | — |
| **Batched-query variance reduction** | ❌ | — |
| **Sign-based update** | ❌ | — |
| **Quantized / low-precision** | ❌ | — |

The Top 3 below cover three of the five empty axes.

### 🥇 #1 — ZO-Muon  (axis: **2D matrix structure / spectral**)
- **Paper:** Lang et al., *Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization* — [arXiv:2602.17155](https://arxiv.org/abs/2602.17155)
- **Official code:** `github.com/OPTML-Group/ZO-Muon`.
- **Reported gain:** **beats both MeZO and LOZO on wall-clock**. OPT-13B / SST-2: 92.5% in 2h33m / 40k queries vs MeZO 91.4% in 3h37m. +25.1% over MeZO on ViT-B / CIFAR-100.
- **What it adds:** Newton–Schulz orthogonalization on a multi-query SPSA estimate inside a per-layer low-rank random subspace (r ∈ {64,128}, Nq ∈ {4,8,16}).
- **Why it's different from everything in the repo:** every existing optimizer treats parameters as a flat vector. ZO-Muon is the first that uses the **2D matrix structure** of weights — orthogonalizing the update spectrally. The Newton–Schulz kernel is also a reusable building block for any future matrix-aware method.
- **Implementation effort:** Medium. ~80 lines over `mezo.py` (per-layer P, multi-query loop, Newton–Schulz `3/2·A − 1/2·A·Aᵀ·A` for 5 iterations, 1D fallback to MeZO).

### 🥈 #2 — ConMeZO  (axis: **perturbation distribution shape**)
- **Paper:** *Constrained-cone MeZO* — [arXiv:2511.02757](https://arxiv.org/abs/2511.02757) (AISTATS 2026), with public code.
- **Reported gain:** **~2× speedup over MeZO**.
- **What it adds:** samples perturbations from a **cone around the momentum direction** rather than isotropic Gaussian — bias the search toward the descent direction.
- **Why it's different from everything in the repo:** every existing optimizer keeps the perturbation distribution isotropic Gaussian and modifies the update rule. ConMeZO is the only one that changes the **distribution itself**. It stacks cleanly on top of any other method (LOZO, Sparse-MeZO, HiZOO, DiZO).
- **Implementation effort:** Low. ~30–50 lines: maintain a unit-norm momentum direction `μ_t`, generate `z ~ N(0,I)`, attenuate the off-axis component by `(1−α)`, renormalize. SPSA two-forward loop unchanged.

### 🥉 #3 — FZOO  (axis: **batched-query variance reduction**)
- **Paper:** *FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning LLMs with Adaptive Batched Forward Passes* — [arXiv:2506.09034](https://arxiv.org/abs/2506.09034)
- **Reported gain:** **~18× fewer forward passes than MeZO**, matching Adam-level convergence rate.
- **What it adds:** batched **one-sided Rademacher** perturbations (multiple `z`'s in one forward) + **std-adaptive learning rate** that scales with the empirical noise level of the batch.
- **Why it's different from everything in the repo:** every existing optimizer uses Nq = 1 query per step with Gaussian perturbations. FZOO does Nq ≫ 1 with Rademacher and adapts LR to the *measured* variance. It also makes the forward batching itself a primitive — the existing methods can be re-cast under it.
- **Implementation effort:** Medium. Refactor the SPSA loop to do one batched forward over `Nq` perturbations (Rademacher `{±1}`); compute empirical std of the `Nq` finite differences; LR scales as `lr / max(σ_hat, ε)`.

### What we dropped from the previous Top 3 and why

| Dropped | Reason |
|---------|--------|
| **HELENE** (was #1) | Lives on the **same axis as HiZOO** (diagonal-Hessian preconditioning). Adds clipping + A-GNB estimator on top, but reviewers and your own ablation table will see it as "HiZOO++". Keep as an honorable mention — useful for a 2nd-order ablation later, but not a top-3 reproduction. |

---

## 3. Suggested execution order

1. **ConMeZO** first — easiest reproduction (~30–50 lines), fastest end-to-end validation of the new-method scaffolding. Confirm the 2× claim against your MeZO baseline before committing to harder reproductions.
2. **ZO-Muon** second — official OPTML-Group code accelerates this. Newton–Schulz becomes a reusable primitive.
3. **FZOO** last — biggest claimed payoff (18×) but requires refactoring the SPSA loop to batch perturbations and add the std-adaptive LR. Save until the suite plumbing is rock-solid.

---

## 4. After reproduction — combinations worth trying

Once all three are reproduced and benchmarked head-to-head against the six existing optimizers, the natural next step is combinations. These crosses are **unpublished** as of late 2025 (per the research scan), and each combines two empty axes:

| Combination | Axes crossed | Why it's interesting |
|-------------|--------------|---------------------|
| **ZO-Muon × ConMeZO** | matrix structure × distribution shape | Cone-sampled perturbations aligned with the spectral structure that Newton–Schulz already exploits. |
| **ZO-Muon × FZOO** | matrix structure × batched variance reduction | ZO-Muon already needs Nq > 1; FZOO formalizes the batching and adds adaptive LR. Natural fit. |
| **LOZO × ZO-Muon** | (existing) low-rank × matrix structure | Newton–Schulz on LOZO's low-rank `UVᵀ` factors instead of full matrices — dramatically cheaper NS. |
| **ConMeZO × LOZO** | distribution shape × (existing) low-rank | Cone sampling *inside* LOZO's low-rank subspace — biases the random V resample toward descent. |
| **FZOO × Sparse-MeZO** | batched variance reduction × (existing) sparse | Batched Rademacher applied through the sparse mask — variance reduction where it matters most. |

---

## 5. Honorable mentions (revisit later)

Strong methods that did not make the top 3 — most redundant with axes the repo already covers, or against the project's "near-inference memory" theme.

| Method | arXiv | Axis | Why it was passed over |
|--------|-------|------|----------------------|
| **HELENE** | [2411.10696](https://arxiv.org/abs/2411.10696) | diagonal Hessian | Up to 20× over MeZO but **same axis as HiZOO**. Useful later for a 2nd-order ablation (HiZOO vs HELENE same scaffolding); not a top-3 variety pick. |
| **JAGUAR SignSGD** | [2506.04430](https://arxiv.org/abs/2506.04430) | sign-based | Sign + coordinate momentum, public code. Genuinely new axis — promote if we want a 4th method. |
| **QuZO** | [2502.12346](https://arxiv.org/abs/2502.12346) | quantized / low-precision | Genuinely new axis (INT4/INT8/FP8 with stochastic rounding) but requires quantization infra in `train.py`. |
| **MeZO-SVRG** | [2404.08080](https://arxiv.org/abs/2404.08080) | variance reduction (full-batch) | Adds substantial memory — against the "near-inference memory" theme. |
| **P-GAP** | [2510.18228](https://arxiv.org/abs/2510.18228) | data-driven low-rank | Same axis as LOZO. |
| **SubZero** | [2410.08989](https://arxiv.org/abs/2410.08989) | per-layer low-rank | Same axis as LOZO. |
| **LOREN** | [2511.07971](https://arxiv.org/abs/2511.07971) | low-rank curvature | Same axis as HiZOO + LOZO crossed. |
| **AdaMeZO** | [2605.00650](https://arxiv.org/abs/2605.00650) | implicit Adam moments | ICML 2026, no public benchmark yet — watch this one. |
| **TeZO** | [2501.19057](https://arxiv.org/abs/2501.19057) | temporal low-rank | Niche. |
