import torch
import numpy as np
import math
from collections import deque

class PZOTrainerHelper:
    def __init__(self, args, optimizer, lr_scheduler):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.sliding_window_length = 14
        self.sliding_window = deque(maxlen=self.sliding_window_length)
        self.coefficients = []
        self.momentum_fb = 0.9
        self.momentum_fb_min = 0.0
        self.momentum_fb_max = 1.0
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
        
        # Sample the random seed for sampling z
        random_seed = np.random.randint(1000000000)
        loss, o0, grad_last = self.pzo_forward(model, inputs_copy, need_grad=True)
        self.pzo_perturb_parameters(random_seed, 1)
        loss1, o1, _ = self.pzo_forward(model, inputs, need_grad=False)
        self.grad_last = grad_last[0]
        self.pzo_perturb_parameters(random_seed, -1)
        o = o1 - o0
        self.o_last = o0
        self.sliding_window.append((random_seed, o))
        return loss1

    def pzo_update(self, model):
        args = self.args
        dot_products = []
        random_seeds = []
        # o: (bs,seq,d_hidden)
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
