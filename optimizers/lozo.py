import torch
from torch.optim.optimizer import Optimizer

class LOZO(Optimizer):
    """
    Low-rank Zeroth-Order SGD (LOZO) optimizer.
    Mathematically aligned 100% with the official paper:
    "Enhancing Zeroth-order fine-tuning for language models with low-rank structures"
    """
    def __init__(self, params, lr=1e-3, eps=1e-3, r=4, nu=50):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eps=eps, r=r, nu=nu)
        super(LOZO, self).__init__(params, defaults)

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
        
        # 1. Apply perturbation for F+
        for group in self.param_groups:
            nu = group['nu']
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                
                if p.dim() >= 2:
                    if state['step'] % nu == 0:
                        # Resample V_l
                        V_dim = p.numel() // p.size(0)
                        state['V'] = torch.randn(V_dim, r, dtype=p.dtype).to(p.device)
                    
                    # Sample U_l
                    state['U'] = torch.randn(p.size(0), r, dtype=p.dtype).to(p.device)
                    
                    # Perturb X_l <- X_l + eps * U_l V_l^T
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                else:
                    # 1D parameters (bias, LayerNorm): fall back to standard MeZO perturbation
                    state['Z'] = torch.randn_like(p)
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
        
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                
                if p.dim() >= 2:
                    # Reset to original X_l and update X_l <- X_l - lr * c * U_l V_l^T
                    net_alpha = eps - (lr * c)
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
    Mathematically aligned 100% with the official paper:
    "Enhancing Zeroth-order fine-tuning for language models with low-rank structures"
    """
    def __init__(self, params, lr=1e-3, eps=1e-3, r=4, nu=50, beta=0.9):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eps=eps, r=r, nu=nu, beta=beta)
        super(LOZOM, self).__init__(params, defaults)

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
        
        # 1. Apply perturbation for F+
        for group in self.param_groups:
            nu = group['nu']
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    if p.dim() >= 2:
                        state['N'] = torch.zeros(p.size(0), r, device=p.device, dtype=p.dtype)
                    else:
                        state['N_1d'] = torch.zeros_like(p)
                
                if p.dim() >= 2:
                    if state['step'] % nu == 0:
                        V_dim = p.numel() // p.size(0)
                        if 'V' in state:
                            V_old = state['V']
                            V_new = torch.randn(V_dim, r, dtype=p.dtype).to(p.device)
                            
                            # Project momentum onto the new subspace: N = N_old @ (V_old.T @ V_new) / V_dim
                            state['N'] = (state['N'] @ (V_old.T @ V_new)) / V_dim
                            state['V'] = V_new
                        else:
                            state['V'] = torch.randn(V_dim, r, dtype=p.dtype).to(p.device)
                    
                    # Sample U_l
                    state['U'] = torch.randn(p.size(0), r, dtype=p.dtype).to(p.device)
                    
                    # Perturb X_l <- X_l + eps * U_l V_l^T
                    p_view = p.view(p.size(0), -1)
                    p_view.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                else:
                    # 1D parameters: fall back to standard MeZO perturbation
                    state['Z'] = torch.randn_like(p)
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
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
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
