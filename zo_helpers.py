import torch
import math
import numpy as np
from collections import deque

# =========================================================================
# SubZero Helper 
# Exact mathematical extraction from SubZero/large_models/trainer.py
# =========================================================================
def fast_svd_method_v2(w_shape, device, dtype, rank=8):
    U, _ = torch.linalg.qr(torch.randn((w_shape[0], rank), device=device))
    U = U.to(dtype).contiguous()
    V, _ = torch.linalg.qr(torch.randn((w_shape[1], rank), device=device))
    Vt = V.to(dtype).T.contiguous()
    return U, Vt
    
def reshape_matrix(integer):
    factor1, factor2 = 1, integer
    for i in range(1, int(math.sqrt(integer)) + 1):
        if int(integer / i) == integer / i:
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
        args = self.args
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
                param.grad = None

        self.zo_random_seed = np.random.randint(1000000000)
        self.zo_subspace_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        assert args.q == 1, "only support q=1 for the memory efficiency."
        for _ in range(args.q):
            if self.args.perturbation_mode == "one_side":
                self.zo_subspace_perturb_parameters(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:
                self.zo_subspace_perturb_parameters(scaling_factor=-2)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                self.zo_subspace_perturb_parameters(scaling_factor=1)

        assert getattr(self.args, 'gradient_accumulation_steps', 1) == 1
        return loss1

    def zo_subspace_update(self, model):
        args = self.args
        torch.manual_seed(self.zo_random_seed)
        for name, param, U, V in self.named_parameters_to_optim:
            if len(torch.squeeze(param.data).shape) == 2:    
                z0 = torch.normal(mean=0, std=1, size=(args.gauss_rank, args.gauss_rank), device=param.data.device, dtype=param.data.dtype)
                z = (U @ z0 @ V * math.sqrt(param.data.numel() / z0.numel())).view(param.data.shape).to(param.data.dtype)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            param.grad = self.projected_grad * z
            self.optimizer.step()
            param.grad = None
            
        self.update_steps += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

# =========================================================================
# PseuZO Helper
# Exact mathematical extraction from PseuZO/pzo_trainer.py
# =========================================================================
class PZOTrainerHelper:
    def __init__(self, args, optimizer, lr_scheduler):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.sliding_window_length = 14
        self.sliding_window = deque(maxlen=self.sliding_window_length)
        self.coefficients = []
        self.momentum_fb = 0.9
        self.zo_random_seed = None
        self.grad_last = None
        self.o_last = None
        self.named_parameters_to_optim = []

    def _get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def reset_momentum_fb(self, momentum_fb):
        self.momentum_fb = momentum_fb
        self.coefficients = []
        for i in range(self.sliding_window_length):
            if i == 0:
                self.coefficients.append(1.0)
            else:
                self.coefficients = [co * self.momentum_fb for co in self.coefficients]
                self.coefficients.append(1.0)

    def Random_noise(self, size, device, dtype, type='Gaussian'):
        if type == 'Gaussian':
            return torch.normal(mean=0, std=1, size=size, device=device, dtype=dtype)
        elif type == 'Rademacher':
            return torch.randint(0, 2, size=size, device=device, dtype=dtype) - 1
        else:
            raise NotImplementedError

    def pzo_perturb_parameters(self, random_seed, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        with torch.no_grad():
            for name, param in self.named_parameters_to_optim:
                z = self.Random_noise(size=param.data.size(), device=param.data.device, dtype=param.data.dtype, type='Gaussian')
                param.data = param.data + scaling_factor * z * self.args.zo_eps

    def pzo_forward(self, model, inputs, need_grad=False):
        model.eval()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        if need_grad:
            with torch.enable_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                logits.requires_grad_(True)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
                grad_last = torch.autograd.grad(loss, logits)[0].detach()
                o = logits.detach()
                return loss.detach(), o, grad_last
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
                return loss.detach(), logits.detach(), None

    def pzo_step(self, model, inputs):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        inputs_copy = inputs.copy()
        
        random_seed = np.random.randint(1000000000)
        loss, o0, grad_last = self.pzo_forward(model, inputs_copy, need_grad=True)
        self.pzo_perturb_parameters(random_seed, 1)
        loss1, o1, _ = self.pzo_forward(model, inputs, need_grad=False)
        self.grad_last = grad_last
        self.pzo_perturb_parameters(random_seed, -1)
        o = o1 - o0
        self.o_last = o0
        self.sliding_window.append((random_seed, o))
        return loss1

    def pzo_update(self, model):
        args = self.args
        dot_products = []
        random_seeds = []
        for seed, do in self.sliding_window:
            b1, seq1, _ = do.shape
            b2, seq2, _ = self.grad_last.shape
            min_b = min(b1, b2)
            min_seq = min(seq1, seq2)
            
            do_sliced = do[:min_b, -min_seq:, :]
            grad_sliced = self.grad_last[:min_b, -min_seq:, :]
            
            dot_product = torch.sum(do_sliced * grad_sliced, dim=(-3, -2, -1))
            dot_products.append(dot_product)
            random_seeds.append(seed)
        coefficients = [co * dot.item() / self.args.zo_eps for (co, dot) in zip(self.coefficients[-len(dot_products):], dot_products)]
        with torch.no_grad():
            for i, (project_value, random_seed) in enumerate(zip(coefficients, random_seeds)):
                torch.manual_seed(random_seed)
                for name, param in self.named_parameters_to_optim:
                    z = self.Random_noise(size=param.data.size(), device=param.data.device, dtype=param.data.dtype, type='Gaussian')
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        param.data = param.data - self._get_learning_rate() * (project_value * z + args.weight_decay * param.data)
                    else:
                        param.data = param.data - self._get_learning_rate() * (project_value * z) 
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

# =========================================================================
# LOZO Helper
# Exact mathematical extraction from LOZO/large_models/LOZOtrainer.py
# =========================================================================
class LOZOTrainerHelper:
    def __init__(self, args, optimizer, lr_scheduler):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = 0
        self.v = {}
        self.zo_random_seed = None
        self.projected_grad = None
        self.named_parameters_to_optim = []

    def _get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def random_gaussian_matrix(self, m, n, device, dtype, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
        random_matrix = torch.randn(m, n, device=device, dtype=dtype)
        return random_matrix

    @torch.no_grad()
    def lowrank_zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        args = self.args
        step = self.step
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                if step % args.step_interval == 0:
                    v = torch.randn(param.data.size(1), args.rank_r, device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v
                else:
                    v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank_r, device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * (u@v.t()) * self.args.zo_eps
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
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
    def lowrank_zo_step(self, model, inputs):
        args = self.args
        if hasattr(self, 'step'):
            self.step += 1
        else:
            self.step = 0
            self.v = {}

        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        self.zo_random_seed = np.random.randint(1000000000)

        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        self.lowrank_zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        assert getattr(self.args, 'gradient_accumulation_steps', 1) == 1
        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        return loss1

    @torch.no_grad()
    def lowrank_zo_update(self):
        args = self.args
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            if param.data.ndim >= 2:
                v = self.v[name]
                u = self.random_gaussian_matrix(m=param.data.size(0), n=args.rank_r, device=param.data.device, dtype=param.data.dtype)

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * (u@v.t()) + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * (u@v.t()))
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
