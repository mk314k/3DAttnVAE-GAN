"""
Variational Autoencoder Architecture using Multihead Attentions
Author: mk314k
"""
from typing import Tuple
import torch
from torch import nn
from models.attention import R3DAttention


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=3, img_channel=1, out_channel=256):
        """
        Args:
            patch_size (int): kernel size of the first and last convolution
            img_channel (int): channels of the input image
            out_channel (int): embedding dimension of each patch
        """
        super().__init__()
        mid_channel = out_channel // 8
        mid_kernel = 2 * patch_size - 1
        self.stride = mid_kernel * patch_size
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(img_channel, mid_channel, kernel_size=patch_size, padding=patch_size // 2),
            nn.GELU(),
            nn.Conv2d(mid_channel, 4 * mid_channel, kernel_size=mid_kernel, stride=mid_kernel),
            nn.GELU(),
            nn.Conv2d(4 * mid_channel, out_channel, kernel_size=patch_size, stride=patch_size),
        )

    def num_patches(self, image_size: Tuple[int, int]) -> int:
        """
        Args:
            image_size (tuple[int, int]): (height, width) of the input image

        Returns:
            int: number of patches produced for an image of that size
        """
        height, width = image_size
        return (height // self.stride) * (width // self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch*view, channel, height, width)

        Returns:
            torch.Tensor: (batch*view, patch, embedding)
        """
        out = self.patch_embedding(x)
        return out.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block: attention followed by an MLP, both residual
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = R3DAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch, tokens, embedding)

        Returns:
            torch.Tensor: (batch, tokens, embedding)
        """
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class R3DEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    Encodes multiple views of an object into a single latent sample.
    Patches of each view attend to each other, then the views attend to each other
    and are averaged into one object embedding.
    """
    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channel=1,
        image_size=(256, 192),
        embedding_dim=64,
        embedding_kernel=3,
        attention_head=4,
        num_layers=2,
        latent_dim=1024,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbed(
            img_channel=in_channel,
            out_channel=embedding_dim,
            patch_size=embedding_kernel
        )
        num_patches = self.patch_embedding.num_patches(image_size)
        if num_patches == 0:
            raise ValueError(
                f"image_size {image_size} is smaller than the patch stride "
                f"{self.patch_embedding.stride}"
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.patch_attention = nn.Sequential(
            *[TransformerBlock(embedding_dim, attention_head) for _ in range(num_layers)]
        )
        self.view_attention = TransformerBlock(embedding_dim, attention_head)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Sequential(nn.Linear(embedding_dim, 256), nn.GELU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): (batch, view, channel, height, width)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: latent sample (batch, latent_dim), or the
            mean in eval mode, and the KL divergence to N(0, I) averaged over the batch
        """
        if x.dim() != 5:
            raise ValueError("input must be of shape (batch, view, channel, height, width)")
        b, v, c, h, w = x.shape
        x = self.patch_embedding(x.reshape(b * v, c, h, w))
        x = torch.cat([self.cls_token.expand(b * v, -1, -1), x], dim=1)
        x = self.patch_attention(x + self.pos_embedding)

        views = self.norm(x[:, 0]).reshape(b, v, -1)
        obj = self.view_attention(views).mean(dim=1)

        hidden = self.fc(obj)
        mu_val = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z_val = mu_val
        if self.training:
            z_val = mu_val + torch.exp(0.5 * logvar) * torch.randn_like(mu_val)
        kl_val = -0.5 * (1 + logvar - mu_val.pow(2) - logvar.exp()).sum(dim=1).mean()
        return z_val, kl_val
