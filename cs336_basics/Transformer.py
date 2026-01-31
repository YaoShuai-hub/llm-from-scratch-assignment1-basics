import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from einops import rearrange, repeat

# [Source: 518] 辅助函数：截断正态分布初始化
def init_weights(module: nn.Module, std: float = 1.0, lower: float = -3.0, upper: float = 3.0):
    if hasattr(module, "weight"):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=lower, b=upper)

class Linear(nn.Module):
    """
    [Source: 526] Implement a Linear class that inherits from torch.nn.Module.
    Performs y = Wx. No bias.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # [Source: 541] construct and store your parameter as W (not W^T)
        # Shape: (out_features, in_features) for consistency with F.linear, 
        # though mathematical notation in doc uses column vectors.
        # However, PyTorch F.linear expects (out_features, in_features).
        # Doc says "store your parameter as W (not W^T) for memory ordering reasons".
        # Let's verify Eq (3): y = xW^T (row vector convention).
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        # [Source: 515] Initialization
        # std = sqrt(2 / (in + out))
        std = math.sqrt(2.0 / (in_features + out_features))
        init_weights(self, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Source: 522] y = Wx (conceptually), but usually xW^T in PyTorch implementation
        return x @ self.weight.T

class Embedding(nn.Module):
    """
    [Source: 552] Implement the Embedding class.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # [Source: 567] store with d_model being the final dimension
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        
        # [Source: 516] Initialization: N(0, 1) truncated at [-3, 3]
        init_weights(self, std=1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # [Source: 550] indexing into embedding matrix
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """
    [Source: 598] Root Mean Square Layer Normalization.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # [Source: 587] Learnable gain parameter g (weight)
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Source: 589] Upcast to float32
        input_dtype = x.dtype
        x_fp32 = x.float()
        
        # [Source: 587] RMS(a) = sqrt(mean(a^2) + eps)
        # Calculate mean over the last dimension
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(mean_square + self.eps)
        
        # [Source: 584] a * (a / RMS(a)) * g
        # Normalize and apply gain
        x_norm = x_fp32 * rms
        
        # [Source: 597] Downcast back and multiply weight
        return (x_norm.to(input_dtype) * self.weight)

class SwiGLUFeedForward(nn.Module):
    """
    [Source: 647] SwiGLU feed-forward network.
    FFN(x) = W2(SiLU(W1x) * W3x)
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # [Source: 639] W1, W3: d_model -> d_ff; W2: d_ff -> d_model
        # No bias [Source: 628]
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Source: 638] SwiGLU(x) = W2(SiLU(W1x) * W3x)
        # Use torch.nn.functional.silu
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        # [Source: 630] Implement SiLU manually: x * sigmoid(x)
        # [Source: 650] explicitly permits torch.sigmoid
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        
        return self.w2(silu_w1_x * w3_x)

class RotaryPositionalEmbedding(nn.Module):
    """
    [Source: 669] Rotary Position Embeddings (RoPE).
    Implementation Note: Uses "adjacent pairs" strategy consistent with 
    Vaswani/RoFormer and the assignment description (block diagonal 2x2).
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        
        # [Source: 657] theta_i = 1 / (theta ^ (2k/d))
        # 0, 2, 4...
        indices = torch.arange(0, d_k, 2, device=device).float()
        freqs = 1.0 / (theta ** (indices / d_k))
        
        t = torch.arange(max_seq_len, device=device).float()
        # Shape: (seq_len, d_k/2)
        freqs = torch.outer(t, freqs)
        
        # [Fix] 关键修改：从切半连接 (cat) 改为相邻重复 (repeat)
        # 我们需要 [theta_0, theta_0, theta_1, theta_1, ...]
        # 之前的 cat 是 [theta_0, theta_1, ..., theta_0, theta_1, ...]
        freqs_cis = repeat(freqs, 's d -> s (d 2)')
        
        self.register_buffer("cos_cached", freqs_cis.cos())
        self.register_buffer("sin_cached", freqs_cis.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # [Fix] 确保类型安全
        token_positions = token_positions.long()
        
        # 查表
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # [Fix] 增强广播逻辑
        # x: (Batch, Heads, Seq, Dim) -> 4D
        # cos: (Batch, Seq, Dim) -> 3D (如果 positions 是 2D)
        #      (Seq, Dim) -> 2D (如果 positions 是 1D)
        
        if x.ndim == 4 and cos.ndim == 3:
            cos = cos.unsqueeze(1) # (B, 1, S, D)
            sin = sin.unsqueeze(1)
        elif x.ndim == 4 and cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0) # (1, 1, S, D)
            sin = sin.unsqueeze(0).unsqueeze(0)
        
        return (x * cos) + (self._rotate_half(x) * sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Fix] 相邻元素交换：
        Input: [x0, x1, x2, x3, ...]
        Output: [-x1, x0, -x3, x2, ...]
        """
        # 1. 把最后一维拆成 (d/2, 2)
        # x.shape[:-1] 获取前面的所有维度
        # view 是零拷贝操作，非常高效
        x = x.view(x.shape[:-1] + (x.shape[-1] // 2, 2))
        
        # 2. 解包：x1 是偶数索引 (0, 2...), x2 是奇数索引 (1, 3...)
        x1, x2 = x.unbind(dim=-1)
        
        # 3. 组合成 [-x2, x1] 并恢复形状
        # stack 后 shape 为 (..., d/2, 2)
        # flatten (view) 回 (..., d)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    [Source: 690] Implement softmax with numerical stability trick.
    Subtract max before exp.
    """
    # [Source: 694] Subtract max for stability
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - max_val
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    [Source: 711] Scaled Dot-Product Attention.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    d_k = q.size(-1)
    
    # [Source: 702] Compute scores: QK^T / sqrt(d_k)
    # q: (batch, heads, seq, d_k), k: (batch, heads, seq, d_k) -> scores: (batch, heads, seq, seq)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # [Source: 709] Apply mask: add -inf where mask is False
        # Ensure mask is broadcastable
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    # [Source: 697] Softmax
    attn_weights = softmax(scores, dim=-1)
    
    # [Source: 701] Multiply by V
    output = torch.matmul(attn_weights, v)
    return output

class MultiHeadSelfAttention(nn.Module):
    """
    [Source: 743] Causal Multi-Head Self-Attention.
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope: RotaryPositionalEmbedding, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # [Source: 747] d_k = d_v = d_model / h
        self.head_dim = d_model // num_heads
        self.rope = rope
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # [Source: 730] W_q, W_k, W_v projections
        # Can be combined [Source: 733], but implementing separately for clarity/standard practice
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Projections
        # Shape: (batch, seq, d_model) -> (batch, seq, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # 2. Split heads
        # Shape: (batch, seq, num_heads, head_dim)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)

        # [Fix] 使用传入的位置，如果为 None 则生成默认位置
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            # 如果是默认位置 (seq_len)，RoPE 内部会自动广播到 (batch, heads, seq, dim)

        # 3. Transpose for attention (batch, heads, seq, head_dim)
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')
        
        # 4. Apply RoPE [Source: 740]
        # Apply to q and k only. Treat head dim as batch dim (independent).
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        # 5. Causal Masking [Source: 738]
        # Allow i to attend to j <= i.
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        # We want True where we DO attend, so invert triu (upper triangle is future)
        # However, scaled_dot_product_attention doc says: "mask value of True ... sum to 1"
        # wait, usually mask is True for KEEP.
        # [Source: 705] "True ... indicates query i does attend to key j"
        # So we want the lower triangle to be True.
        causal_mask = ~mask # Invert upper triangle
        
        # 6. Scaled Dot-Product Attention
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        
        # 7. Concatenate heads and output projection
        # (batch, heads, seq, head_dim) -> (batch, seq, d_model)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        
        return self.w_o(attn_output)
    
class TransformerBlock(nn.Module):
    """
    [Source: 757] Pre-norm Transformer Block.
    y = x + MHA(RMSNorm(x))
    z = y + FFN(RMSNorm(y))
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope: RotaryPositionalEmbedding, device=None, dtype=None):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope, device=device, dtype=dtype)
        
        self.rms_norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Source: 754] Sublayer 1: MHA
        # Residual connection
        x = x + self.mha(self.rms_norm1(x))
        
        # [Source: 751] Sublayer 2: FFN
        x = x + self.ffn(self.rms_norm2(x))
        
        return x

class TransformerLM(nn.Module):
    """
    [Source: 764] The Full Transformer Language Model.
    """
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        d_ff: int, 
        rope_theta: float = 10000.0,
        attn_pdrop: float = 0.0, # Not explicitly mentioned but good practice, keeping 0 as per doc silence
        device=None, 
        dtype=None
    ):
        super().__init__()
        
        # [Source: 767] Determine dimensionality of position embedding matrix
        # RoPE module is shared across layers [Source: 665]
        # theta defaults to 10000 per Source 1124
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model//num_heads, max_seq_len=context_length, device=device)
        
        # [Source: 369] Token Embeddings
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # [Source: 375] Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        # [Source: 380] Final RMSNorm
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # [Source: 381] Output Head (Linear)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        
        # 1. Embeddings
        x = self.token_embedding(input_ids)
        
        # 2. Transformer Blocks
        for layer in self.layers:
            x = layer(x)
            
        # 3. Final Norm
        x = self.final_norm(x)
        
        # 4. Output Head -> Logits
        logits = self.lm_head(x)
        
        return logits