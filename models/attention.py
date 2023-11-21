"""
This Module includes general implementation of Multiheaded Attention
"""
import torch
from torch import nn
from fancy_einsum import einsum


class R3DAttention(nn.Module):
    """
    R3DAttention module performs multi-head attention computation on 3D data.

    Args:
        hidden_size (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads to use.
        dropout (float, optional): Dropout rate to prevent overfitting (default: 0.1).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout=0.1):
        super().__init__()

        # Calculating the size of each attention head
        head_size = int(hidden_size / num_heads)
        self.n_head = num_heads
        self.head_size = head_size

        # Projection layers for Q, K, and V
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Perform multi-head scaled dot-product attention on the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch*view, patch, embedding).
            mask (torch.Tensor, optional): Mask tensor for masking attention scores (default: None).

        Returns:
            torch.Tensor: Output tensor after attention computation (batch*view, patch, embedding).
        """
        b, p, _ = x.shape

        # Projecting input into query, key, and value representations
        qkv = self.qkv_proj(x).reshape(b, p, 3, self.n_head, self.head_size)
        q, k, v = qkv.chunk(dim=2)

        # Calculating attention scores
        attn_score = einsum("b n pq s, b n pk s->b n pq pk", q, k) / (self.head_size ** 0.5)

        # Applying optional mask to attention scores
        if mask is not None:
            attn_score -= mask

        # Computing attention probabilities and apply dropout
        attn_prob = attn_score.softmax(dim=-1)
        attn_prob = self.attn_dropout(attn_prob)

        # Weighted sum of values using attention probabilities
        z = einsum("b n pq pk, b n pk s ->b pq n s", attn_prob, v)
        z = z.reshape((z.shape[0], z.shape[1], -1))

        # Projecting back to the original space and applying residual dropout
        out = self.output_proj(z)
        out = self.resid_dropout(out)

        return out
