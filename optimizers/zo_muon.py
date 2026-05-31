import math

import torch
from torch.optim.optimizer import Optimizer


# Quintic Newton-Schulz coefficients from Keller Jordan's reference Muon
# implementation (https://kellerjordan.github.io/posts/muon/). The official
# ZO-Muon code (OPTML-Group/ZO-Muon, llm/optimizers.py) uses these verbatim.
# Iteration: A <- a*A + b*(A A^T) A + c*(A A^T)^2 A.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def _newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the matrix sign.

    Operates on a 2D tensor; orthogonalises the spectrum so the output is
    close to U V^T where G = U S V^T. Computed in float32 because the
    iteration is unstable in fp16/bf16. The input is normalised by its
    Frobenius norm first.

    Uses the quintic coefficients (3.4445, -4.7750, 2.0315) from Keller
    Jordan's Muon reference; the official ZO-Muon paper code uses the same.
    """
    assert G.dim() == 2, f"_newton_schulz_5 expects 2D tensor, got {G.dim()}D"
    orig_dtype = G.dtype
    A = G.float()
    # Transpose for tall matrices to keep the smaller dimension last —
    # numerically more stable and matches the official Muon implementation.
    transposed = False
    if A.size(0) > A.size(1):
        A = A.t()
        transposed = True

    A = A / (A.norm() + 1e-7)
    # Quintic iteration: A <- a*A + b*(A A^T) A + c*(A A^T)^2 A.
    for _ in range(steps):
        AAT = A @ A.t()
        AAT2 = AAT @ AAT
        A = _NS_A * A + _NS_B * (AAT @ A) + _NS_C * (AAT2 @ A)

    if transposed:
        A = A.t()
    return A.to(orig_dtype)


class ZOMuon(Optimizer):
    """
    Zeroth-Order Muon (ZO-Muon).
    arXiv:2602.17155 — "Powering Up Zeroth-Order Training via Subspace
    Gradient Orthogonalization". Reference code:
    https://github.com/OPTML-Group/ZO-Muon, scripts/lowdim.sh.

    For each 2D weight of shape (out, in):
        P            ∈ R^{in x r}      sampled Gaussian then QR-orthogonalised;
                                       refreshed every refresh_T steps.
        for k = 1..Nq:
            U_k      ∈ R^{out x r}     fresh Gaussian per query.
            Z_k      = U_k P^T         rank-r perturbation.
            theta   <- theta + eps Z_k;  L_+
            theta   <- theta - 2 eps Z_k; L_-
            theta   <- theta +   eps Z_k  (restore)
            c_k      = (L_+ - L_-) / (2 eps)
            G_low   += (1/Nq) c_k U_k
        G            = G_low P^T                ∈ R^{out x in}
        O            = NewtonSchulz5(G)         (cast to fp32 internally)
        M           <- momentum M + O           (Muon momentum buffer)
        theta       <- theta - lr M

    For 1D weights (biases, LayerNorms) we fall back to plain MeZO averaged
    over the same Nq queries — this matches the official repo's behaviour.

    DDP synchronisation: P, U_k, and the 1D z's are all generated from the
    deterministic per-param seed (`param_seed + k`), so every rank produces
    bit-identical perturbations. Newton-Schulz is element-wise deterministic
    once the input is identical.

    Args:
        lr         (float): learning rate. Paper default 1e-2 (much larger
            than MeZO because the update is orthogonalised, not magnitude-
            scaled).
        eps        (float): perturbation scale.
        r          (int):   subspace rank. Paper defaults: 64 or 128.
        Nq         (int):   number of multi-query SPSA probes per step
            (paper defaults 2-8; 4 is a good balance).
        ns_steps   (int):   number of Newton-Schulz iterations (fixed at 5
            in the paper; exposed for ablation).
        refresh_T  (int):   how often P is re-sampled (paper default 100).
        momentum   (float): Muon momentum coefficient on the orthogonalised
            update (paper default 0.95).
        seed       (int):   base seed for synchronized RNG.
    """

    def __init__(self, params, lr=1e-2, eps=1e-3, r=64, Nq=4,
                 ns_steps=5, refresh_T=50, momentum=0.0,
                 lr_1d=1e-7, seed=42):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if r < 1:
            raise ValueError(f"r must be >= 1: {r}")
        if Nq < 1:
            raise ValueError(f"Nq must be >= 1: {Nq}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1: {ns_steps}")
        if refresh_T < 1:
            raise ValueError(f"refresh_T must be >= 1: {refresh_T}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1): {momentum}")
        if lr_1d < 0.0:
            raise ValueError(f"Invalid lr_1d: {lr_1d}")
        defaults = dict(lr=lr, eps=eps, r=r, Nq=Nq, ns_steps=ns_steps,
                        refresh_T=refresh_T, momentum=momentum,
                        lr_1d=lr_1d, seed=seed)
        super().__init__(params, defaults)
        self._step_count = 0

    # ---- helpers -----------------------------------------------------------

    def _maybe_refresh_P(self, p, state, group, generator):
        """Refresh the per-layer projection matrix P every refresh_T steps."""
        r         = group['r']
        refresh_T = group['refresh_T']
        if p.dim() < 2:
            return  # 1D fallback path doesn't use P

        in_dim = p.numel() // p.size(0)
        need_refresh = ('P' not in state) or (state['step'] % refresh_T == 0)
        if need_refresh:
            # Generate Gaussian then QR -> orthonormal columns in R^{in x r}.
            # QR in float32 for numerical stability (LAPACK quirks in bf16/fp16).
            G = torch.randn(in_dim, r, dtype=torch.float32, device=p.device,
                            generator=generator)
            Q, _ = torch.linalg.qr(G, mode='reduced')
            state['P'] = Q.to(p.dtype)

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("ZOMuon requires a closure that returns the loss.")

        eps  = self.defaults['eps']
        Nq   = int(self.defaults['Nq'])
        seed = self.defaults['seed']

        # Lazy-init state and refresh P (per-layer subspace).
        for group in self.param_groups:
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                if 'generator' not in state:
                    state['generator'] = torch.Generator(device=p.device)

                # Re-seed the per-param generator with the base seed for this
                # step so P refreshes are bit-identical across ranks.
                param_id   = getattr(p, 'param_id', 0)
                param_seed = seed + state['step'] + param_id * 1000003
                state['generator'].manual_seed(param_seed)

                # Allocate the gradient accumulators.
                if p.dim() >= 2:
                    self._maybe_refresh_P(p, state, group, state['generator'])
                    state['G_low'] = torch.zeros(p.size(0), r,
                                                 dtype=p.dtype, device=p.device)
                    if 'momentum_buf' not in state:
                        # Keep Muon's momentum buffer in fp32: fp16 loses ~10
                        # bits per update at typical weight magnitudes, which
                        # corrupts the long-term EMA we feed Newton-Schulz.
                        state['momentum_buf'] = torch.zeros(p.shape,
                                                            dtype=torch.float32,
                                                            device=p.device)
                else:
                    state['acc_1d'] = torch.zeros_like(p.data)

        first_loss = None

        # ---- Nq multi-query SPSA probes ----
        for k in range(Nq):
            # Forward perturbation theta <- theta + eps * Z_k.
            for group in self.param_groups:
                r = group['r']
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]

                    # Re-seed for this query: (param_seed + k+1) is independent
                    # from the P-refresh seed (param_seed) above.
                    param_id   = getattr(p, 'param_id', 0)
                    param_seed = seed + state['step'] + param_id * 1000003
                    state['generator'].manual_seed(param_seed + (k + 1) * 17)

                    if p.dim() >= 2:
                        U_k = torch.randn(p.size(0), r, dtype=p.dtype,
                                          device=p.device,
                                          generator=state['generator'])
                        state['U_k'] = U_k
                        # p <- p + eps * U_k P^T
                        p_view = p.view(p.size(0), -1)
                        p_view.addmm_(U_k, state['P'].t(), alpha=eps, beta=1.0)
                    else:
                        z_k = torch.randn(p.size(), dtype=p.dtype,
                                          device=p.device,
                                          generator=state['generator'])
                        state['z_k'] = z_k
                        p.add_(z_k, alpha=eps)

            L_plus = closure()
            if isinstance(L_plus, torch.Tensor):
                L_plus = L_plus.item()

            # Backward perturbation theta <- theta - 2 eps * Z_k.
            for group in self.param_groups:
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]
                    if p.dim() >= 2:
                        p_view = p.view(p.size(0), -1)
                        p_view.addmm_(state['U_k'], state['P'].t(),
                                      alpha=-2.0 * eps, beta=1.0)
                    else:
                        p.add_(state['z_k'], alpha=-2.0 * eps)

            L_minus = closure()
            if isinstance(L_minus, torch.Tensor):
                L_minus = L_minus.item()

            c_k = (L_plus - L_minus) / (2.0 * eps)
            if first_loss is None:
                first_loss = L_plus

            # Restore theta (queries are independent, do not fuse the update)
            # and accumulate into the gradient buffer.
            for group in self.param_groups:
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]
                    if p.dim() >= 2:
                        # Restore: p += eps * U_k P^T.
                        p_view = p.view(p.size(0), -1)
                        p_view.addmm_(state['U_k'], state['P'].t(),
                                      alpha=eps, beta=1.0)
                        # Accumulate: G_low += (c_k / Nq) * U_k.
                        state['G_low'].add_(state['U_k'], alpha=c_k / Nq)
                        del state['U_k']
                    else:
                        p.add_(state['z_k'], alpha=eps)
                        state['acc_1d'].add_(state['z_k'], alpha=c_k / Nq)
                        del state['z_k']

        # ---- Lift, orthogonalise, momentum, apply ----
        self._step_count += 1
        for group in self.param_groups:
            lr   = group['lr']
            beta = group['momentum']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]

                if p.dim() >= 2:
                    # G_full = G_low @ P^T  in R^{out x in}
                    G_full = state['G_low'] @ state['P'].t()
                    G_full = G_full.view_as(p.data)
                    ns_steps = group.get('ns_steps', 5)
                    O = _newton_schulz_5(G_full, steps=ns_steps)
                    # Muon RMS-match rescale: keep the per-element step
                    # magnitude calibrated to Adam-equivalent across shapes.
                    # Without this the orthogonalised update has element RMS
                    # ~1/sqrt(min(out,in)) and the effective per-coordinate
                    # step is ~30x too small at OPT-1.3B's typical 2048x2048
                    # / 8192x2048 matrices, leaving ~2pp on the table.
                    # Reference: Keller Jordan's Muon (kellerjordan.github.io)
                    # and OPTML-Group/ZO-Muon's llm/optimizers.py.
                    out_dim = p.size(0)
                    in_dim  = p.numel() // p.size(0)
                    rms_scale = max(1.0, math.sqrt(out_dim / in_dim)) * 0.2
                    if beta > 0.0:
                        state['momentum_buf'].mul_(beta).add_(O.float(),
                                                              alpha=rms_scale)
                        p.add_(state['momentum_buf'].to(p.dtype), alpha=-lr)
                    else:
                        p.add_(O, alpha=-lr * rms_scale)
                    del state['G_low']
                else:
                    # 1D fallback: plain ZO-SGD with the Nq-averaged g_hat.
                    # CRITICAL: 2D weights go through Newton-Schulz which
                    # orthogonalises updates to element magnitude ~1/sqrt(in_dim),
                    # so the muon `lr` (e.g. 1e-2) is appropriately small. 1D
                    # weights have NO such normalization -- raw `acc_1d` has
                    # element magnitude ~O(c_k) ~ O(1). Applying lr=1e-2 to that
                    # would thrash biases / layer norms ~5 orders of magnitude
                    # too aggressively (plain MeZO uses lr=1e-7 for this form).
                    # We use a separate `lr_1d` (default 1e-7) for 1D updates.
                    p.add_(state['acc_1d'], alpha=-group['lr_1d'])
                    del state['acc_1d']

                state['step'] += 1

        return first_loss if first_loss is not None else 0.0
