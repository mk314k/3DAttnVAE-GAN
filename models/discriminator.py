"""
This module includes 3D Discriminator module
"""
import torch
from torch import nn

def conv3d(in_channel:int, out_channel:int):
    """
    Conv3d structure with default hyperparameters
    Args:
        in_channel (int): channel of image before this layer
        out_channel (int): channel of image after this layer

    Returns:
        nn.Module: Conv3d Module
    """
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        nn.LeakyReLU(0.2, inplace=True),
    )


class R3Discriminator(nn.Module):
    """
    Discriminator Module
    """
    def __init__(self):
        super().__init__()
        self.out_channels = 512
        self.out_dim = 4
        self.conv1 = conv3d(1, 64)
        self.conv2 = conv3d(64, 128)
        self.conv3 = conv3d(128, 256)
        self.conv4 = conv3d(256, 512)
        self.out = nn.Linear(512 * self.out_dim * self.out_dim * self.out_dim, 1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x (torch.Tensor): voxel occupancy (batch, 1, 64, 64, 64)

        Returns:
            torch.Tensor: real/fake logit (batch, 1)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.out(x.flatten(1))
