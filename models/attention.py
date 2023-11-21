"""_summary_
"""
import torch
import torch.nn as nn
from fancy_einsum import einsum


class R3DAttention(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout=0.1):
        super(R3DAttention, self).__init__()
        head_size = int(hidden_size / num_heads)
        self.n_head = num_heads
        self.head_size = head_size
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): (batch*view, patch, embedding)

        Returns:
            torch.Tensor: (batch*view, patch, embedding)
        """
        b, p, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, p, 3, self.n_head, self.head_size)
        q, k, v = qkv.chunk(dim=2)
        #b-batch, p-patch, c-constant(3), n-num_heads, s-head_size
        attn_score = einsum("b n pq s, b n pk s->b n pq pk", q, k) / (
            self.head_size**0.5
        )
        if mask:
            attn_score -= mask
        attn_prob = attn_score.softmax(dim=-1)
        attn_prob = self.attn_dropout(attn_prob)
        z = einsum("b n pq pk, b n pk s ->b pq n s", attn_prob, v)
        z = z.reshape((z.shape[0], z.shape[1], -1))
        out = self.output_proj(z)
        out = self.resid_dropout(out)
        return out


# if __name__ == "__main__":
#     attn = R3DAttention(64, 4)
#     def count_par(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(count_par(attn))
