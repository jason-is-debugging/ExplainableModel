

from torch import nn
import torch

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    """Single transformer decoder block with causal self-attention."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)

        return x

