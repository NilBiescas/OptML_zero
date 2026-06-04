import torch
import numpy as np
import math

def fast_svd_method_v2(w_shape, device, dtype, rank=8):
    U, _ = torch.linalg.qr(torch.randn((w_shape[0], rank), device=device))
    U = U.to(dtype).contiguous()
    
    V, _ = torch.linalg.qr(torch.randn((w_shape[1], rank), device=device))
    Vt = V.to(dtype).T.contiguous()
    
    return U, Vt
    
def reshape_matrix(integer):
    factor1, factor2 = 1, integer
    for i in range(1, int(math.sqrt(integer)) + 1):  # range: [1, sqrt(x) + 1)
        if int(integer / i) == integer / i:  # i is factor
            if integer / i - i < factor2 - factor1:
                factor1, factor2 = i, integer / i
    return factor1, int(factor2)

class SubZeroTrainerHelper:
    def __init__(self, args, optimizer, lr_scheduler):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.p_state = {}
        self.update_steps = 0
        self.state = type('State', (), {'global_step': 0})()
        self.zo_random_seed = None
        self.projected_grad = None
        self.named_parameters_to_optim = []

    @torch.no_grad()
    def zo_subspace_perturb_parameters(self, random_seed=None, scaling_factor=1):
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for _, param, U, V in self.named_parameters_to_optim:
            if len(U.shape) == 1:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)    
                param.data = param.data + scaling_factor * z * self.args.zo_eps
                
            elif len(U.shape) == 2:
                z = torch.normal(mean=0, std=1, size=(U.shape[1], V.shape[0]), device=param.data.device, dtype=param.data.dtype)
                z = (U @ z @ V * math.sqrt(param.data.numel() / z.numel())).view(param.data.shape)
                param.data = param.data + scaling_factor * z * self.args.zo_eps 

    def zo_forward(self, model, inputs):
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            loss = outputs.loss
        return loss.detach()

    @torch.no_grad()
    def zo_subspace_step(self, model, inputs):
        """
        Estimate gradient by subzero. Return the loss from f(theta + z)
        """
        args = self.args
                
        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:

                if len(torch.squeeze(param.data).shape) == 2:
                    if self.state.global_step == 0:
                        self.p_state[name] = {'U': torch.zeros(param.data.size(0), args.gauss_rank), 
                                                'V': torch.zeros(args.gauss_rank, param.data.size(1))}
                  
                    p_state = self.p_state[name]          
                        
                    if self.state.global_step % args.update_interval == 0:
                        if args.mode in ['lora', 'prefix', 'prompt']:
                            w_shape = reshape_matrix(param.data.numel())
                            print(w_shape)
                            U, V = fast_svd_method_v2(w_shape=w_shape, device=param.device, dtype=param.data.dtype, rank=args.gauss_rank)
                        else:
                            U, V = fast_svd_method_v2(w_shape=param.data.shape, device=param.device, dtype=param.data.dtype, rank=args.gauss_rank)
                      
                        p_state['U'] = U
                        p_state['V'] = V
                        
                    U = p_state['U']
                    V = p_state['V']  
                    
                    self.named_parameters_to_optim.append((name, param, U, V))
                else:
                    self.named_parameters_to_optim.append((name, param, torch.Tensor([1.]), torch.Tensor([1.])))
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_subspace_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency."
        for _ in range(args.q):
            if self.args.perturbation_mode == "one_side":
                self.zo_subspace_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_subspace_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

                # Reset model back to its parameters at start of step
                self.zo_subspace_perturb_parameters(scaling_factor=1)

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1

    def zo_subspace_update(self, model):
        args = self.args
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(self.zo_random_seed)
        for name, param, U, V in self.named_parameters_to_optim:
            # Resample z
            if len(torch.squeeze(param.data).shape) == 2:    
                z0 = torch.normal(mean=0, std=1, size=(args.gauss_rank, args.gauss_rank), device=param.data.device, dtype=param.data.dtype)
                z = (U @ z0 @ V * math.sqrt(param.data.numel() / z0.numel())).view(param.data.shape).to(param.data.dtype)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                             dtype=param.data.dtype)

            param.grad = self.projected_grad * z
            self.optimizer.step()  # will only update grad that is not None.
            param.grad = None  # avoid further update.
            
        self.update_steps += 1
        if self.update_steps % 1000 == 0:
            print('model update', self.update_steps)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
