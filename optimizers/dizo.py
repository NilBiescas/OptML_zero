import torch
from torch.optim.optimizer import Optimizer


class DiZO(Optimizer):
    """
    Divergence-driven Zeroth-Order (DiZO) optimizer.
    NeurIPS 2025: "Harmony in Divergence: Towards Fast, Accurate, and
    Memory-efficient Zeroth-order LLM Fine-tuning" (arXiv:2502.03304)

    Two-phase update:
      Phase 1 (every step): MeZO-style full-rank ZO update on all parameters.
      Phase 2 (every `kappa` steps): for each Q/V attention layer, ZO-optimize
        a per-layer scalar gamma_l, then apply the divergence-driven projection:
            theta_l = theta0_l + (gamma_l / ||delta_l||) * delta_l
        where delta_l = theta_l - theta0_l is the displacement from the
        pre-trained anchor. This rescales each layer's divergence so that
        updates match first-order magnitudes per layer.

    Hyperparameters (paper Table 1 defaults):
        lr       : main ZO learning rate (1e-6 full-param, 1e-4 LoRA)
        eps      : main perturbation scale (1e-3 full-param, 1e-2 LoRA)
        eps_proj : perturbation scale for gamma ZO sub-problem (0.05–0.1)
        kappa    : projection update interval in steps (50–400)
        tau      : clip ratio gamma/||delta|| to [1-tau, 1+tau] (0.1–0.3)
        j_proj   : inner ZO iterations per projection update (5 recommended)
        seed     : base seed for deterministic multi-GPU synchronization
    """

    # Layer name substrings that identify Q and V attention projections.
    # Covers Qwen, LLaMA, Mistral, OPT, GPT-2, Falcon, etc.
    _QV_PATTERNS = ('q_proj', 'v_proj', 'query', 'value', 'wq', 'wv')

    def __init__(self, params, lr=1e-7, eps=1e-3, eps_proj=1e-1,
                 kappa=100, tau=0.2, j_proj=5, lr_proj=1e-3, seed=42):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps, eps_proj=eps_proj,
                        kappa=kappa, tau=tau, j_proj=j_proj,
                        lr_proj=lr_proj, seed=seed)
        super(DiZO, self).__init__(params, defaults)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_qv(self, p) -> bool:
        """True when p is a 2-D Q or V attention weight matrix."""
        name = getattr(p, 'param_name', '')
        return p.dim() >= 2 and any(pat in name for pat in self._QV_PATTERNS)

    def _init_anchors(self):
        """Lazily clone pre-trained weights for Q/V layers on first call."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and self._is_qv(p):
                    state = self.state[p]
                    if 'anchor' not in state:
                        state['anchor'] = p.data.clone()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("DiZO requires a closure that returns the loss.")

        self._init_anchors()

        eps  = self.defaults['eps']
        seed = self.defaults['seed']

        # ---- Phase 1: MeZO-style ZO update on all parameters ----

        # Forward perturbation: p <- p + eps * z
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0

                param_id   = getattr(p, 'param_id', 0)
                param_seed = seed + state['step'] + param_id * 1000003

                if 'generator' not in state:
                    state['generator'] = torch.Generator(device=p.device)
                state['generator'].manual_seed(param_seed)

                # z stored temporarily; freed after the update below
                state['z'] = torch.randn(p.shape, dtype=p.dtype,
                                         device=p.device,
                                         generator=state['generator'])
                p.add_(state['z'], alpha=eps)

        loss_plus = closure()
        if isinstance(loss_plus, torch.Tensor):
            loss_plus = loss_plus.item()

        # Backward perturbation: p <- p - 2*eps*z  (net: p_orig - eps*z)
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                p.add_(self.state[p]['z'], alpha=-2.0 * eps)

        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()

        c = (loss_plus - loss_minus) / (2.0 * eps)
        self._step_count += 1

        # Restore and update: p <- p_orig - lr*c*z
        # Currently p = p_orig - eps*z; add (eps - lr*c)*z to get p_orig - lr*c*z
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                p.add_(state['z'], alpha=eps - lr * c)
                del state['z']
                state['step'] += 1

        # ---- Phase 2: Projection step every kappa steps ----
        if self._step_count % self.defaults['kappa'] == 0:
            self._projection_step(closure)

        return loss_plus

    # ------------------------------------------------------------------
    # Projection step (DiZO's core innovation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _projection_step(self, closure):
        """
        ZO-optimize a per-Q/V-layer scalar ratio alpha_l (= gamma_l / ||delta_l||)
        and apply:
            theta_l = theta0_l + clip(alpha_l, 1-tau, 1+tau) * delta_l

        Reparameterising the paper's gamma_l as the ratio alpha_l keeps the ZO
        perturbation `eps_proj * u` on the same scale as alpha (~1), regardless
        of how small ||delta_l|| is — without this, eps_proj=0.1 against gamma~
        ||delta||~5e-3 makes ratio ≈ 1 ± 20, which destroys the model during
        the closure() probes and turns c_gamma into pure noise.
        """
        eps_proj = self.defaults['eps_proj']
        j_proj   = self.defaults['j_proj']
        tau      = self.defaults['tau']
        seed     = self.defaults['seed']
        lr_proj  = self.defaults['lr_proj']  # separate from main `lr`: alpha
                                             # lives near 1, not weight scale

        qv_params = [
            p
            for group in self.param_groups
            for p in group['params']
            if p.requires_grad and self._is_qv(p) and 'anchor' in self.state[p]
        ]
        if not qv_params:
            return

        L      = len(qv_params)
        device = qv_params[0].device
        dtype  = qv_params[0].dtype

        # Displacement from anchor (kept constant during the inner ZO loop;
        # only the per-layer scaling alpha changes).
        deltas = [p.data - self.state[p]['anchor'] for p in qv_params]

        # alpha_l = ratio gamma_l / ||delta_l||. Identity projection at start.
        alpha = torch.ones(L, dtype=dtype, device=device)

        proj_gen      = torch.Generator(device=device)
        proj_seed_base = seed + self._step_count * 77777

        for j in range(j_proj):
            proj_gen.manual_seed(proj_seed_base + j)
            u = torch.randn(L, dtype=dtype, device=device, generator=proj_gen)

            # Forward perturbation on alpha: theta = theta0 + alpha_+ * delta
            for i, p in enumerate(qv_params):
                a = (alpha[i] + eps_proj * u[i]).clamp(1.0 - tau, 1.0 + tau)
                p.data.copy_(self.state[p]['anchor'] + a * deltas[i])

            l_plus = closure()
            if isinstance(l_plus, torch.Tensor):
                l_plus = l_plus.item()

            # Backward perturbation on alpha
            for i, p in enumerate(qv_params):
                a = (alpha[i] - eps_proj * u[i]).clamp(1.0 - tau, 1.0 + tau)
                p.data.copy_(self.state[p]['anchor'] + a * deltas[i])

            l_minus = closure()
            if isinstance(l_minus, torch.Tensor):
                l_minus = l_minus.item()

            # Restore weights to current alpha (so closure-side state is clean)
            for i, p in enumerate(qv_params):
                p.data.copy_(self.state[p]['anchor'] + alpha[i] * deltas[i])

            # ZO gradient on alpha, update, then clip back into [1-tau, 1+tau]
            c_alpha = (l_plus - l_minus) / (2.0 * eps_proj)
            alpha.add_(u, alpha=-lr_proj * c_alpha)
            alpha.clamp_(1.0 - tau, 1.0 + tau)

        # Final projection write-back to the actual model weights
        for i, p in enumerate(qv_params):
            p.data.copy_(self.state[p]['anchor'] + alpha[i] * deltas[i])
