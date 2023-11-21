"""
Variational Autoencoder Architecture
Author: mk314k
"""
import torch
from torch import nn
from attention import R3DAttention


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size: int, img_channel: int, embedding_channel: int):
        """_summary_

        Args:
            patch_size (int): _description_
            in_channel (int): _description_
            out_channel (int): _description_
        """

        super().__init__()
        self.patch_embedding = nn.Conv2d(img_channel, embedding_channel, kernel_size=patch_size)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): (batch*view, channel, width, height)

        Returns:
            torch.Tensor: (batch*view, patch, embedding)
        """
        out = self.patch_embedding(x).view(x.size(0), -1, self.num_patches).permute(0, 2, 1)
        return out


class R3DEncoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        in_channel=1,
        num_patches=128,
        embedding_dim=64,
        embedding_kernel=3,
        attention_head=4,
        latent_dim=1024
    ):
        super().__init__()
        self.patch_embedding = PatchEmbed(
            img_channel=in_channel,
            embedding_channel=embedding_dim,
            patch_size=embedding_kernel
        )
        self.pos_embedding = nn.Embedding(num_patches, embedding_dim)
        self.batch_attention = R3DAttention(embedding_dim, attention_head)
        # self.view_attention = R3DAttention(2 * conv1_channel, 8)
        self.fc = nn.Linear(embedding_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.dist = torch.distributions.Normal(0, 1)
        self.dist.loc = self.N.loc.cuda()
        self.dist.scale = self.N.scale.cuda()
        self.kl_val = 0

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): input tensor
            Each image can have multiple view

        Returns:
            torch.Tensor: _description_
        """
        # apply encoder network to input image
        # assert(len(x.shape) == 5, "input must be of shape (batch, view, channel, width, height)")
        b, v, c, w, h = x.shape
        x = x.view(b * v, c, w, h)
        x = self.patch_embedding(x)
        x = x + self.pos_embedding() #fix it
        x = x + self.patch_attention(x)
        x = self.mlp(x)
        mu_val = self.fc_mu(x)
        sigma = self.fc_logvar(x).exp()
        z_val = mu_val + sigma * self.dist.sample(mu_val.shape)
        self.kl_val = (sigma ** 2 + mu_val**2 - sigma.log() - 1 / 2).sum()
        return z_val
