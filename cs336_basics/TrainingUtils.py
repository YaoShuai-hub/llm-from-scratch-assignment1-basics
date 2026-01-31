import torch
import math
from typing import Optional, Iterable, Tuple, Union

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    [Source: 824] Compute the cross entropy loss.
    l_i = -log(softmax(o_i)[x_{i+1}])
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        
    Returns:
        loss: scalar tensor (average over batch and sequence)
    """
    # Flatten batch and sequence dimensions for easier processing
    # logits: (N, vocab_size), targets: (N,)
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # [Source: 826] Subtract the largest element for numerical stability.
    # log_softmax(x)_i = x_i - max(x) - log(sum(exp(x_j - max(x))))
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0] # (N, 1)
    logits_shifted = logits - max_logits
    
    # [Source: 827] Cancel out log and exp whenever possible.
    # log(softmax(x)) = logits_shifted - log(sum(exp(logits_shifted)))
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True)) # (N, 1)
    log_probs = logits_shifted - log_sum_exp
    
    # Select the log_prob corresponding to the target class
    # Use torch.gather or advanced indexing
    # We need log_probs[i, targets[i]] for each i
    target_log_probs = log_probs[torch.arange(targets.size(0)), targets]
    
    # [Source: 828] Return the average across the batch.
    # Cross entropy is negative log likelihood
    return -target_log_probs.mean()


class AdamW(torch.optim.Optimizer):
    """
    [Source: 927] Implements AdamW algorithm.
    """
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        [Source: 849] Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # [Source: 920] g <- gradient
                grad = p.grad.data
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # [Source: 917] m <- 0, v <- 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # [Source: 920] m <- beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # [Source: 920] v <- beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # [Source: 921] Compute adjusted alpha for iteration t (Bias correction)
                # alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # [Source: 922] theta <- theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # [Source: 923] Decoupled weight decay: theta <- theta - alpha * lambda * theta
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
    

def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    [Source: 969] Cosine annealing learning rate schedule with warmup.
    """
    # [Source: 964] Warm-up
    if t < T_w:
        return (t / T_w) * alpha_max
    
    # [Source: 965] Cosine annealing
    if T_w <= t <= T_c:
        cosine_term = math.cos((t - T_w) / (T_c - T_w) * math.pi)
        return alpha_min + 0.5 * (1 + cosine_term) * (alpha_max - alpha_min)
    
    # [Source: 965] Post-annealing
    return alpha_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    """
    [Source: 979] Clip gradients to max_norm.
    """
    # Convert generator to list to iterate multiple times
    params = list(params)
    
    # [Source: 975] Compute L2 norm of all gradients
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm_sq)
    
    # [Source: 975-976] Scale down if norm > M
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale_factor)

import numpy as np

def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [Source: 994] Sample a batch of data.
    """
    # [Source: 990] Choose random starting indices
    # Valid indices are [0, len(data) - context_length - 1]
    # We need indices such that idx + context_length + 1 <= len(data)
    # np.random.randint(low, high) 的 high 是开区间（不包含）
    max_idx = len(data) - context_length
    ix = np.random.randint(0, max_idx, (batch_size,))
    
    # [Source: 988] Slice x and y
    # x: input sequence, y: target sequence (shifted by 1)
    x = torch.stack([torch.from_numpy((data[i : i + context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in ix])
    
    # [Source: 995] Move to device
    if device == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        
    return x, y

import os

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: Union[str, os.PathLike]):
    """
    [Source: 1026] Save model and optimizer state.
    """
    # [Source: 1021] Use state_dict()
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    # [Source: 1023] torch.save
    torch.save(state, out)

def load_checkpoint(src: Union[str, os.PathLike], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    [Source: 1036] Load checkpoint and return iteration.
    """
    # [Source: 1038] torch.load
    checkpoint = torch.load(src, map_location='cpu') # Load to cpu first
    
    # [Source: 1038] load_state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']