"""
This module includes code for Generating 3D images
"""
import torch
from torch import nn


def deconv3d(in_channel: int, out_channel: int) -> nn.Module:
    """
    ConvTranspose3d that doubles each spatial dimension
    """
    return nn.Sequential(
        nn.ConvTranspose3d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm3d(out_channel),
        nn.ReLU(),
    )


class R3DGenerator(nn.Module):
    """
    Decodes a latent vector into a 64x64x64 voxel grid of occupancy logits
    """
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 64 * 4 * 4 * 4)
        self.conv1 = deconv3d(64, 32)
        self.conv2 = deconv3d(32, 16)
        self.conv3 = deconv3d(16, 8)
        self.conv4 = nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): (batch, z_dim)

        Returns:
            torch.Tensor: occupancy logits (batch, 1, 64, 64, 64)
        """
        out = self.linear(z).view(-1, 64, 4, 4, 4)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.conv4(out)
