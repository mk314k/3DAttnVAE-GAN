"""
This is an optional wrapper for VAE_GAN Model
"""
import torch
from encoder import R3DEncoder
from generator import R3DGenerator
from discriminator import R3Discriminator

class R3D:
    """
    The main model
    """
    def __init__(self, # pylint: disable=too-many-arguments
        in_channel=1,
        num_patches=128,
        embedding_dim=64,
        embedding_kernel=3,
        attention_head=4,
        latent_dim=1024
    ):
        self.encoder = R3DEncoder(
            in_channel, num_patches, embedding_dim, embedding_kernel, attention_head, latent_dim
        )
        self.generator = R3DGenerator(latent_dim)
        self.discriminator = R3Discriminator()
        self.enc = None
        self.gan = None
        self.disc_true = None
        self.disc_false = None

    def __call__(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        self.enc = self.encoder(x)
        self.gan = self.generator(self.enc)
        self.disc_true = self.discriminator(y.reshape(1, *y.shape))
        self.disc_false = self.discriminator(self.gan)

        return self.gan

    def vae_parameters(self):
        """
        Returns:
            list of parameters for encoding 2d images and generating 3d images
        """
        return list(self.encoder.parameters()) + list(self.generator.parameters())

    def gan_parameters(self):
        """
        Returns:
            list of parameters for 3d image generation and discrimination
        """
        return list(self.generator.parameters()) + list(self.discriminator.parameters())
