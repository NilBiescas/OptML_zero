import torch
import math
from torch.optim.optimizer import Optimizer

class LOZO(Optimizer):
    """
    Low-rank Zeroth-Order SGD (LOZO) optimizer.
    With fully synchronized multi-GPU random generators to support 
    correct mathematical behavior across distributed nodes without DDP drift.
    """
    def __init__(self, params, lr=1e-3, eps=1e-3, r=4, nu=50, seed=42):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eps=eps, r=r, nu=nu, seed=seed)
        super(LOZO, self).__init__(params, defaults)
        self._generators = {}

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise RuntimeError("LOZO optimizer requires a closure that computes the loss")

        eps = self.defaults['eps']
        seed = self.defaults.get('seed', 42)
        
        # 1. Apply perturbation for F+
        for group in self.param_groups:
            nu = group['nu']
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                
                # Retrieve the deterministic parameter ID injected by the training loop
                param_id = getattr(p, 'param_id', 0)
                
                # Deterministic seed combining base seed, parameter ID, and step number
                # ensures perfect alignment across different GPU processes
                param_seed = seed + state['step'] + param_id * 1000003
                if p not in self._generators:
                    self._generators[p] = torch.Generator(device=p.device)
                generator = self._generators[p]
                generator.manual_seed(param_seed)
                
                if p.dim() >= 2:
                    if state['step'] % nu == 0 or 'V' not in state:
                        # Resample V_l
                        V_dim = p.numel() // p.size(0)
                        state['V'] = torch.randn(V_dim, r, dtype=p.dtype, device=p.device, generator=generator)
                    
                    # Sample U_l
                    state['U'] = torch.randn(p.size(0), r, dtype=p.dtype, device=p.device, generator=generator)
                    
                    # Perturb X_l <- X_l + eps * U_l V_l^T
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                else:
                    # 1D parameters (bias, LayerNorm): fall back to standard MeZO perturbation
                    state['Z'] = torch.randn(p.size(), dtype=p.dtype, device=p.device, generator=generator)
                    p.add_(state['Z'], alpha=eps)
        
        # Calculate F+
        loss_plus = closure()
        if isinstance(loss_plus, torch.Tensor):
            loss_plus = loss_plus.item()
            
        # 2. Apply perturbation for F-
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                
                if p.dim() >= 2:
                    # X_l <- X_l - 2 * eps * U_l V_l^T
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=-2 * eps, beta=1.0)
                else:
                    p.add_(state['Z'], alpha=-2 * eps)
        
        # Calculate F-
        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()
            
        # 3. Reset parameters and update
        c = (loss_plus - loss_minus) / (2 * eps)
        
        # DEBUG PRINTING (Once every 10 steps or first step)
        step_idx = getattr(self, '_step_count', 0)
        # if step_idx % 10 == 0:
        #     print(f"[LOZO Optimizer Debug] step: {step_idx} | loss_plus: {loss_plus:.8f} | loss_minus: {loss_minus:.8f} | diff: {loss_plus - loss_minus:.8e} | c: {c:.8e}")
        self._step_count = step_idx + 1
        
        for group in self.param_groups:
            lr = group['lr']
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                
                if p.dim() >= 2:
                    # Reset to original X_l and update X_l <- X_l - lr * c * U_l V_l^T / r
                    net_alpha = eps - (lr * c / r)
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=net_alpha, beta=1.0)
                else:
                    net_alpha = eps - (lr * c)
                    p.add_(state['Z'], alpha=net_alpha)
                
                state['step'] += 1

        return loss_plus


class LOZOM(Optimizer):
    """
    Low-rank Zeroth-Order SGD with Momentum (LOZO-M) optimizer.
    With fully synchronized multi-GPU random generators to support
    correct mathematical behavior across distributed nodes without DDP drift.
    """
    def __init__(self, params, lr=1e-3, eps=1e-3, r=4, nu=50, beta=0.9, seed=42):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eps=eps, r=r, nu=nu, beta=beta, seed=seed)
        super(LOZOM, self).__init__(params, defaults)
        self._generators = {}

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise RuntimeError("LOZOM optimizer requires a closure that computes the loss")

        eps = self.defaults['eps']
        seed = self.defaults.get('seed', 42)
        
        # 1. Apply perturbation for F+
        for group in self.param_groups:
            nu = group['nu']
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                
                if p.dim() >= 2 and 'N' not in state:
                    state['N'] = torch.zeros(p.size(0), r, device=p.device, dtype=p.dtype)
                elif p.dim() < 2 and 'N_1d' not in state:
                    state['N_1d'] = torch.zeros_like(p)
                
                # Retrieve the deterministic parameter ID injected by the training loop
                param_id = getattr(p, 'param_id', 0)
                
                # Deterministic seed combining base seed, parameter ID, and step number
                # ensures perfect alignment across different GPU processes
                param_seed = seed + state['step'] + param_id * 1000003
                if p not in self._generators:
                    self._generators[p] = torch.Generator(device=p.device)
                generator = self._generators[p]
                generator.manual_seed(param_seed)
                
                if p.dim() >= 2:
                    if state['step'] % nu == 0 or 'V' not in state:
                        V_dim = p.numel() // p.size(0)
                        if 'V' in state:
                            V_old = state['V']
                            # Generate new V using the synchronized generator
                            V_new = torch.randn(V_dim, r, dtype=p.dtype, device=p.device, generator=generator)
                            
                            # Project momentum onto the new subspace: N = N_old @ (V_old.T @ V_new) / sqrt(V_dim)
                            # Using sqrt(V_dim) mathematically preserves momentum scale, avoiding collapse!
                            state['N'] = (state['N'] @ (V_old.T @ V_new)) / math.sqrt(V_dim)
                            state['V'] = V_new
                        else:
                            state['V'] = torch.randn(V_dim, r, dtype=p.dtype, device=p.device, generator=generator)
                    
                    # Sample U_l using the synchronized generator
                    state['U'] = torch.randn(p.size(0), r, dtype=p.dtype, device=p.device, generator=generator)
                    
                    # Perturb X_l <- X_l + eps * U_l V_l^T
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                else:
                    # 1D parameters: fall back to standard MeZO perturbation using the synchronized generator
                    state['Z'] = torch.randn(p.size(), dtype=p.dtype, device=p.device, generator=generator)
                    p.add_(state['Z'], alpha=eps)
        
        # Calculate F+
        loss_plus = closure()
        if isinstance(loss_plus, torch.Tensor):
            loss_plus = loss_plus.item()
            
        # 2. Apply perturbation for F-
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                
                if p.dim() >= 2:
                    # X_l <- X_l - 2 * eps * U_l V_l^T
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=-2 * eps, beta=1.0)
                else:
                    p.add_(state['Z'], alpha=-2 * eps)
        
        # Calculate F-
        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()
            
        # 3. Reset parameters and update momentum
        c = (loss_plus - loss_minus) / (2 * eps)
        
        # DEBUG PRINTING (Once every 10 steps or first step)
        step_idx = getattr(self, '_step_count', 0)
        # if step_idx % 10 == 0:
        #     print(f"[LOZOM Optimizer Debug] step: {step_idx} | loss_plus: {loss_plus:.8f} | loss_minus: {loss_minus:.8f} | diff: {loss_plus - loss_minus:.8e} | c: {c:.8e}")
        self._step_count = step_idx + 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                
                if p.dim() >= 2:
                    # Reset to original X_l
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                    
                    # Update momentum: N_l = beta * N_l + (1 - beta) * c * U_l
                    state['N'].mul_(beta).add_(state['U'], alpha=(1 - beta) * c)
                    
                    # Update X_l <- X_l - lr * N_l V_l^T
                    p_view.addmm_(state['N'], state['V'].T, alpha=-lr, beta=1.0)
                else:
                    # Reset to original X_l
                    p.add_(state['Z'], alpha=eps)
                    
                    # Update momentum
                    state['N_1d'].mul_(beta).add_(state['Z'], alpha=(1 - beta) * c)
                    
                    # Update parameter
                    p.add_(state['N_1d'], alpha=-lr)
                
                state['step'] += 1

        return loss_plus
