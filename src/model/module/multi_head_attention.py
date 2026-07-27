"""Multi-Head Self-Attention module.

Key idea (one-liner):
    For each token, we measure how "relevant" it is to every other token,
    then take a weighted sum of the value vectors using those relevance scores.

Multi-head means the model can look at "relevance" from several perspectives
at once — e.g. one head focuses on syntactic structure, another on semantics,
another on coreference — and the results are concatenated at the end.

Notation used throughout:
    B  = batch size  (how many sentences are processed in parallel)
    T  = seq_len     (number of tokens per sentence)
    H  = num_heads   (how many attention heads run in parallel)
    d_k = d_model / H  (dimensionality of each head)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention layer.

    Args:
        d_model:  Total embedding dimension. Must be divisible by ``num_heads``.
        num_heads: Number of parallel attention heads.
        dropout:   Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Single fused projection: maps (B, T, d_model) -> (B, T, 3 * d_model)
        # The 3*d_model slice is then split into Q, K, V along the last axis.
        # We keep one big Linear rather than three separate ones to reduce
        # the number of parameters and improve memory locality.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Projects the concatenated multi-head output back to d_model.
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    # ----------------------------------------------------------------------- #
    # forward                                                                 #
    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute multi-head self-attention over the input sequence.

        Args:
            x:    Input tensor of shape (B, T, d_model).
            mask: Optional mask. Shape (B, T) or (T, T).
                  Positions with value 0 are blocked ("cannot attend").
                  Commonly used for:
                    - padding masks (block padding tokens)
                    - future-token masks (decoder causal masking)

        Returns:
            Tensor of shape (B, T, d_model) — the attention-enhanced output.
        """
        B, T, _ = x.shape

        # ------------------------------------------------------------------ #
        # 1. Project input into Query, Key, Value vectors
        # ------------------------------------------------------------------ #
        # qkv shape: (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split the 3*d_model dimension into three separate d_model chunks.
        # Reshape: (B, T, 3*d_model) -> (B, T, 3, H, d_k)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.d_k)

        # Rearrange so that the head dimension comes before the sequence.
        # (B, T, 3, H, d_k) -> (3, B, H, T, d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Unpack into three independent tensors, each (B, H, T, d_k)
        q = qkv[0]   # Query: "what am I looking for?"
        k = qkv[1]   # Key:   "what do I contain?"
        v = qkv[2]   # Value: "what is my actual content?"

        # ------------------------------------------------------------------ #
        # 2. Compute attention scores: (Q · K^T) / sqrt(d_k)
        # ------------------------------------------------------------------ #
        # q @ k.transpose(-2, -1) yields shape (B, H, T, T).
        # Entry [b, h, i, j] = relevance of query i to key j in batch b, head h.
        #
        # Dividing by sqrt(d_k) prevents dot-product values from growing
        # large as d_k increases, which would push softmax into saturated
        # regions where gradients vanish.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ------------------------------------------------------------------ #
        # 3. Apply causal (future-token) mask
        # ------------------------------------------------------------------ #
        # causal_mask[i, j] = True  means "token i must NOT look at token j"
        # We create an upper-triangular matrix (excluding the diagonal):
        #   [[0, 1, 1, ...],
        #    [0, 0, 1, ...],
        #    [0, 0, 0, ...], ...]
        # True entries are where j > i (future positions).
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Masked positions receive -inf so softmax outputs exactly 0 probability.
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # ------------------------------------------------------------------ #
        # 4. Apply external mask (e.g. padding mask)
        # ------------------------------------------------------------------ #
        # mask == 0  →  positions that should be masked out
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # ------------------------------------------------------------------ #
        # 5. Softmax → attention weights (each row sums to 1)
        # ------------------------------------------------------------------ #
        # softmax(scores, dim=-1) normalises over the key dimension (dim T).
        # High score → high weight → the corresponding value contributes more.
        attn_weights = F.softmax(scores, dim=-1)   # (B, H, T, T)
        attn_weights = self.dropout(attn_weights)   # regularisation

        # ------------------------------------------------------------------ #
        # 6. Weighted sum of values
        # ------------------------------------------------------------------ #
        # attn_weights @ v: (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        # For each token, we compute a weighted average of all values,
        # where the weights are the attention scores.
        output = torch.matmul(attn_weights, v)      # (B, H, T, d_k)

        # ------------------------------------------------------------------ #
        # 7. Concatenate heads and project back to d_model
        # ------------------------------------------------------------------ #
        # (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, H * d_k == d_model)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(B, T, self.num_heads * self.d_k)

        # Final linear layer mixes information across heads.
        output = self.out_proj(output)               # (B, T, d_model)

        return output
