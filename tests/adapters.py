from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import math

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# 将上级目录加入 sys.path
sys.path.append(parent_dir)

from cs336_basics.BPE import train_bpe, BPE_Tokenizer

# -----------------------------------------------------------------------------
# 请确保以下 import 路径与你实际保存的文件名一致
# -----------------------------------------------------------------------------
from cs336_basics.Transformer import (
    Linear, 
    Embedding, 
    SwiGLUFeedForward, 
    RMSNorm, 
    MultiHeadSelfAttention, 
    TransformerBlock, 
    TransformerLM, 
    RotaryPositionalEmbedding,
    softmax,
    scaled_dot_product_attention
)

# 假设优化器和Loss函数保存在 Optimizer.py，如果在 Transformer.py 请修改此处
from cs336_basics.TrainingUtils import (
    cross_entropy,
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
    get_batch
)
# -----------------------------------------------------------------------------


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.
    """
    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        layer.weight.copy_(weights)
    
    # 原代码这里是 raise，修正为 return
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
    """
    embedding = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        embedding.weight.copy_(weights)

    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.
    """
    swiglu = SwiGLUFeedForward(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    with torch.no_grad():
        swiglu.w1.weight.copy_(w1_weight)
        swiglu.w2.weight.copy_(w2_weight)
        swiglu.w3.weight.copy_(w3_weight)

    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.
    """
    return scaled_dot_product_attention(Q, K, V, mask)


# 一个假的 RoPE 模块，用于 run_multihead_self_attention
class MockRoPE(nn.Module):
    def forward(self, x, position_ids=None):
        return x

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    This function should not use RoPE.
    """
    # 实例化 MHA，传入 MockRoPE 来禁用旋转
    device = q_proj_weight.device
    dtype = q_proj_weight.dtype
    seq_len = in_features.shape[-2]
    
    mha = MultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=num_heads, 
        max_seq_len=seq_len, 
        rope=MockRoPE(), # 禁用 RoPE
        device=device, 
        dtype=dtype
    )
    
    with torch.no_grad():
        mha.w_q.weight.copy_(q_proj_weight)
        mha.w_k.weight.copy_(k_proj_weight)
        mha.w_v.weight.copy_(v_proj_weight)
        mha.w_o.weight.copy_(o_proj_weight)
        
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    This version of MHA should include RoPE.
    """
    device = q_proj_weight.device
    dtype = q_proj_weight.dtype
    d_k = d_model // num_heads
    
    # 实例化真正的 RoPE
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)
    
    mha = MultiHeadSelfAttention(
        d_model=d_model, 
        num_heads=num_heads, 
        max_seq_len=max_seq_len, 
        rope=rope, 
        device=device, 
        dtype=dtype
    )
    
    with torch.no_grad():
        mha.w_q.weight.copy_(q_proj_weight)
        mha.w_k.weight.copy_(k_proj_weight)
        mha.w_v.weight.copy_(v_proj_weight)
        mha.w_o.weight.copy_(o_proj_weight)
    
    # 注意：之前的 MHA 实现中 forward 只接受 x，RoPE 位置是根据 shape 自动生成的。
    # 如果 adapter 传入了特定的 token_positions，我们需要修改 MHA 的 forward 
    # 或者在这里手动调用。
    # 但根据标准实现，通常 MHA 内部处理位置。为了通过测试，假设 token_positions 
    # 匹配 in_features 的顺序。
    return mha(in_features, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.
    """
    device = in_query_or_key.device
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)
    
    # 我们的 RoPE 实现 forward 接收 (x, token_positions)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.
    """
    device = in_features.device
    dtype = in_features.dtype
    d_k = d_model // num_heads
    
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)
    
    block = TransformerBlock(
        d_model, num_heads, d_ff, max_seq_len, rope, device=device, dtype=dtype
    )
    
    # 映射权重
    new_state_dict = {}
    new_state_dict['mha.w_q.weight'] = weights['attn.q_proj.weight']
    new_state_dict['mha.w_k.weight'] = weights['attn.k_proj.weight']
    new_state_dict['mha.w_v.weight'] = weights['attn.v_proj.weight']
    new_state_dict['mha.w_o.weight'] = weights['attn.output_proj.weight']
    
    new_state_dict['rms_norm1.weight'] = weights['ln1.weight']
    new_state_dict['rms_norm2.weight'] = weights['ln2.weight']
    
    new_state_dict['ffn.w1.weight'] = weights['ffn.w1.weight']
    new_state_dict['ffn.w2.weight'] = weights['ffn.w2.weight']
    new_state_dict['ffn.w3.weight'] = weights['ffn.w3.weight']
    
    block.load_state_dict(new_state_dict, strict=False)
    
    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # 实例化模型，传入 rope_theta
    model = TransformerLM(
        vocab_size, 
        context_length, 
        d_model, 
        num_layers, 
        num_heads, 
        d_ff, 
        rope_theta=rope_theta,
        device=in_indices.device
    )
    
    # 映射权重字典
    new_state_dict = {}
    new_state_dict['token_embedding.weight'] = weights['token_embeddings.weight']
    new_state_dict['final_norm.weight'] = weights['ln_final.weight']
    new_state_dict['lm_head.weight'] = weights['lm_head.weight']
    
    for i in range(num_layers):
        prefix_w = f'layers.{i}'
        prefix_m = f'layers.{i}'
        
        new_state_dict[f'{prefix_m}.rms_norm1.weight'] = weights[f'{prefix_w}.ln1.weight']
        new_state_dict[f'{prefix_m}.rms_norm2.weight'] = weights[f'{prefix_w}.ln2.weight']
        
        new_state_dict[f'{prefix_m}.mha.w_q.weight'] = weights[f'{prefix_w}.attn.q_proj.weight']
        new_state_dict[f'{prefix_m}.mha.w_k.weight'] = weights[f'{prefix_w}.attn.k_proj.weight']
        new_state_dict[f'{prefix_m}.mha.w_v.weight'] = weights[f'{prefix_w}.attn.v_proj.weight']
        new_state_dict[f'{prefix_m}.mha.w_o.weight'] = weights[f'{prefix_w}.attn.output_proj.weight']
        
        new_state_dict[f'{prefix_m}.ffn.w1.weight'] = weights[f'{prefix_w}.ffn.w1.weight']
        new_state_dict[f'{prefix_m}.ffn.w2.weight'] = weights[f'{prefix_w}.ffn.w2.weight']
        new_state_dict[f'{prefix_m}.ffn.w3.weight'] = weights[f'{prefix_w}.ffn.w3.weight']
        
    model.load_state_dict(new_state_dict, strict=False)
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.
    """
    norm = RMSNorm(d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        norm.weight.copy_(weights)
        
    return norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.
    """
    # SiLU = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset... sample language modeling input sequences.
    """
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`.
    """
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
    """
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    """
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Return the learning rate at the given iteration.
    """
    return get_lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    """
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer.
    """
    return BPE_Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer.
    """
    return train_bpe(input_path, vocab_size, special_tokens)