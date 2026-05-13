import torch
from torch.optim.optimizer import Optimizer

class LOZO(Optimizer):
    """
    Low-rank Zeroth-Order SGD (LOZO) optimizer.
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
                
                if state['step'] % nu == 0:
                    # Resample V_l
                    state['V'] = torch.randn(p.size(1) if p.dim() > 1 else 1, r, dtype=p.dtype).to(p.device)
                
                # Sample U_l
                state['U'] = torch.randn(p.size(0), r, dtype=p.dtype).to(p.device)
                
                # Perturb X_l <- X_l + eps * U_l V_l^T
                # Use addmm_ to apply low-rank perturbation directly without instantiating the full-size matrix!
                if p.dim() > 1:
                    p.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                else:
                    p.add_((state['U'] @ state['V'].T).squeeze(-1), alpha=eps)
        
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
                
                # X_l <- X_l - 2 * eps * U_l V_l^T (effectively X_old - eps * U_l V_l^T)
                if p.dim() > 1:
                    p.addmm_(state['U'], state['V'].T, alpha=-2 * eps, beta=1.0)
                else:
                    p.sub_((state['U'] @ state['V'].T).squeeze(-1), alpha=2 * eps)
                
        # Calculate F-
        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()
            
        # 3. Reset parameters and update
        c = (loss_plus - loss_minus) / (2 * eps)
        
        for group in self.param_groups:
            lr = group['lr']
            r = group['r']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                
                # Reset to original X_l and update X_l <- X_l - lr * c * U_l V_l^T / r_l
                net_alpha = eps - (lr * c / r)
                if p.dim() > 1:
                    p.addmm_(state['U'], state['V'].T, alpha=net_alpha, beta=1.0)
                else:
                    p.add_((state['U'] @ state['V'].T).squeeze(-1), alpha=net_alpha)
                
                state['step'] += 1

        return loss_plus


class LOZOM(Optimizer):
    """
    Low-rank Zeroth-Order SGD with Momentum (LOZO-M) optimizer.
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
                    state['N'] = torch.zeros(p.size(0), r, device=p.device, dtype=p.dtype)
                
                if state['step'] % nu == 0:
                    if 'V' in state:
                        V_old = state['V']
                        V_new = torch.randn(p.size(1) if p.dim() > 1 else 1, r, dtype=p.dtype).to(p.device)
                        n_l = p.size(1) if p.dim() > 1 else 1
                        
                        # Project momentum onto the new subspace: N = N_old @ V_old.T @ V_new / n_l
                        state['N'] = (state['N'] @ V_old.T @ V_new) / n_l
                        state['V'] = V_new
                    else:
                        state['V'] = torch.randn(p.size(1) if p.dim() > 1 else 1, r, dtype=p.dtype).to(p.device)
                
                # Sample U_l
                state['U'] = torch.randn(p.size(0), r, dtype=p.dtype).to(p.device)
                
                # Perturb X_l <- X_l + eps * U_l V_l^T
                if p.dim() > 1:
                    p.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                else:
                    p.add_((state['U'] @ state['V'].T).squeeze(-1), alpha=eps)
        
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
                
                # X_l <- X_l - 2 * eps * U_l V_l^T
                if p.dim() > 1:
                    p.addmm_(state['U'], state['V'].T, alpha=-2 * eps, beta=1.0)
                else:
                    p.sub_((state['U'] @ state['V'].T).squeeze(-1), alpha=2 * eps)
                
        # Calculate F-
        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()
            
        # 3. Reset parameters and update momentum
        c = (loss_plus - loss_minus) / (2 * eps)
        
        for group in self.param_groups:
            lr = group['lr']
            r = group['r']
            beta = group['beta']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                state = self.state[p]
                
                # Reset to original X_l
                if p.dim() > 1:
                    p.addmm_(state['U'], state['V'].T, alpha=eps, beta=1.0)
                else:
                    p.add_((state['U'] @ state['V'].T).squeeze(-1), alpha=eps)
                
                # Update momentum: N_l = beta * N_l + (1 - beta) * c * U_l
                state['N'].mul_(beta).add_(state['U'], alpha=(1 - beta) * c)
                
                # Update X_l <- X_l - lr * N_l V_l^T / r_l
                if p.dim() > 1:
                    p.addmm_(state['N'], state['V'].T, alpha=-lr / r, beta=1.0)
                else:
                    p.sub_((state['N'] @ state['V'].T).squeeze(-1), alpha=lr / r)
                
                state['step'] += 1

        return loss_plus
