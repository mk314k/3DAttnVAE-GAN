"""
This Module includes general implementation of Multiheaded Attention
Author:mk314k
"""
import torch
from torch import nn


class R3DAttention(nn.Module):
    """
    R3DAttention module performs multi-head attention computation on Image Patches.

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
        bv, p, _ = x.shape

        # Projecting input into query, key, and value representations
        qkv = self.qkv_proj(x).reshape(bv, p, 3, self.n_head, self.head_size)
        qkv = qkv.permute(2,0,3,1,4) #(3, bv, n, p, h)
        q, k, v = qkv.chunk(3)

        # Calculating attention scores (bv, n, p, p)
        attn_score = q[0].matmul(k[0].transpose(-1,-2))/ (self.head_size ** 0.5)

        # Applying optional mask to attention scores
        if mask is not None:
            attn_score -= mask

        # Computing attention probabilities and apply dropout
        attn_prob = attn_score.softmax(dim=-1) #(bv, n, p, p)
        attn_prob = self.attn_dropout(attn_prob)

        # Weighted sum of values using attention probabilities
        z = attn_prob.matmul(v[0]) #(bv, n, p, h)
        z = z.permute(0, 2, 1, 3) #(bv, p, n, h)
        z = z.reshape((bv, p, -1)) #(bv, n, e)

        # Projecting back to the original space and applying residual dropout
        out = self.output_proj(z) #(bv, n, e)
        out = self.resid_dropout(out)

        return out
