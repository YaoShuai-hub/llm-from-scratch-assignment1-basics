import torch
import numpy as np
import torch.nn.functional as F
import time
import os

# 倒入之前的组件
from Transformer import TransformerLM, softmax
from TrainingUtils import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping, get_batch, save_checkpoint, load_checkpoint

def estimate_loss(model, data, batch_size, context_length, device, eval_iters=100):
    """
    Helper to evaluate loss on a dataset without backprop.
    """
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device)
        with torch.no_grad():
            logits = model(X)
            loss = cross_entropy(logits, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def train(args):
    """
    [Source: 1052] Main training loop.
    args: simple object/dict with hyperparameters
    """
    # 1. Setup Device
    device = args.device
    torch.manual_seed(1337)
    
    # 2. Load Data (Memory Efficient)
    # [Source: 1008] Use np.memmap for large datasets
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode='r')
    
    # 3. Init Model
    # Need to match the class signature we wrote in Section 3
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device
    )
    model.to(device)
    
    # 4. Init Optimizer
    # [Source: 1127] AdamW hyperparameters
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, 
                      betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    
    # 5. Training Loop
    iter_num = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if provided
    if args.resume_from:
        iter_num = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {iter_num}")

    t0 = time.time()
    
    while iter_num < args.max_iters:
        # A. Learning Rate Schedule [Source: 969]
        lr = get_lr_cosine_schedule(iter_num, args.learning_rate, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # B. Get Batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # C. Forward Pass
        # [Source: 360] Forward returns logits
        logits = model(X)
        loss = cross_entropy(logits, Y)
        
        # D. Backward Pass
        optimizer.zero_grad(set_to_none=True) # set_to_none is slightly more efficient
        loss.backward()
        
        # E. Gradient Clipping [Source: 978]
        gradient_clipping(model.parameters(), max_norm=1.0)
        
        # F. Optimizer Step
        optimizer.step()
        
        # G. Logging & Evaluation
        if iter_num % args.eval_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            # Evaluate on train and val
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device)
            print(f"step {iter_num}: train loss {loss.item():.4f}, val loss {val_loss:.4f}, lr {lr:.2e}, time {dt:.2f}s")
            
            # [Source: 1101] Log experiment (simple print here, but could be wandb)
            
            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter_num, os.path.join(args.out_dir, 'ckpt_best.pt'))
                
            # Regular checkpoint
            if iter_num % args.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, iter_num, os.path.join(args.out_dir, f'ckpt_{iter_num}.pt'))

        iter_num += 1

    print("Training finished!")

def generate(model, prompt, max_new_tokens, temperature=1.0, top_p=0.0, device='cpu'):
    """
    [Source: 1084] Generate text from the model.
    """
    model.eval()
    # prompt is (batch, seq_len)
    idx = prompt.to(device)
    
    for _ in range(max_new_tokens):
        # Crop context if too long (model has fixed context window)
        idx_cond = idx if idx.size(1) <= model.rope.max_seq_len else idx[:, -model.rope.max_seq_len:] # assuming rope holds max_seq_len
        
        # Forward
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Take logits at the last position [Source: 1070]
        logits = logits[:, -1, :] / temperature # [Source: 1073] Apply temperature
        
        # [Source: 1077] Top-p Sampling (Nucleus)
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        # Sample
        probs = softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) # (batch, 1)
        
        # Append
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop token check (optional, but good practice per Source 1085)
        # Assuming <|endoftext|> is handled outside or idx check
        
    return idx