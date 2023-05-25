import torch
import torch.nn as nn
from fancy_einsum import einsum
from einops import rearrange

class R3DAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout=0.1):
        super(R3DAttention, self).__init__()
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        head_size = hidden_size // num_heads
        self.head_size = head_size
        self.qkv_proj = nn.Linear(hidden_size,3*hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, cache = None) -> torch.Tensor:
        xshap = x.shape
        if len(xshap)==3:
            x = x.reshape((1,*x.shape))
        b, c, w, h = x.shape
        x = x.permute((0,2,3,1)).reshape((b, -1, c))
        qkv = self.qkv_proj(x)
        n_shape = qkv.shape
        q,k,v = rearrange(qkv,"b s (c n h)-> c b n s h", c=3, n=self.num_heads)
        attn_score = einsum("b n sq h, b n sk h->b n sq sk",q,k)/(self.head_size**0.5)
        mask = 1e4*torch.triu(torch.ones_like(attn_score[0,0]),diagonal=1)
        attn_prob = torch.softmax(attn_score-mask,dim=-1,dtype=x.dtype)#type: ignore
        attn_prob = self.attn_dropout(attn_prob)
        z = einsum("b n sq sk, b n sk h ->b sq n h",attn_prob,v)
        z = torch.reshape(z,(z.shape[0],z.shape[1],-1))
        out = self.output_proj(z)
        out = self.resid_dropout(out)
        out = out.reshape((b, w, h,c)).permute((0,3,1,2))
        return out