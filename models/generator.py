import torch
import torch.nn as nn

class R3DGenerator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, 64 * 4 * 4 * 4)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        out = self.linear(z)
        out = out.view(-1, 64, 4, 4, 4)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out