"""
This is an optional wrapper for VAE_GAN Model
"""
from typing import Tuple
import torch
from torch import nn
from models.encoder import R3DEncoder
from models.generator import R3DGenerator
from models.discriminator import R3Discriminator

class R3D(nn.Module):
    """
    The main model
    """
    def __init__(self, # pylint: disable=too-many-arguments
        in_channel=1,
        image_size=(256, 192),
        embedding_dim=64,
        embedding_kernel=3,
        attention_head=4,
        num_layers=2,
        latent_dim=1024
    ):
        super().__init__()
        self.encoder = R3DEncoder(
            in_channel, image_size, embedding_dim, embedding_kernel,
            attention_head, num_layers, latent_dim
        )
        self.generator = R3DGenerator(latent_dim)
        self.discriminator = R3Discriminator()

    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): views (batch, view, channel, height, width)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: voxel logits (batch, 1, 64, 64, 64)
            and the encoder KL divergence
        """
        z_val, kl_val = self.encoder(x)
        return self.generator(z_val), kl_val

    @torch.no_grad()
    def reconstruct(self, x:torch.Tensor)->torch.Tensor:
        """
        Returns:
            voxel occupancy probabilities (batch, 64, 64, 64)
        """
        logits, _ = self(x)
        return torch.sigmoid(logits).squeeze(1)

    def vae_parameters(self):
        """
        Returns:
            list of parameters for encoding 2d images and generating 3d images
        """
        return list(self.encoder.parameters()) + list(self.generator.parameters())

    def disc_parameters(self):
        """
        Returns:
            list of parameters for discriminating real and generated 3d images
        """
        return list(self.discriminator.parameters())
