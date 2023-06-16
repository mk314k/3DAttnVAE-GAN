import torch
import torch.nn as nn

conv3d = lambda channel: nn.Sequential(
        nn.Conv3d(channel[0], channel[1], kernel_size=4,stride=2, padding=1, bias=False),
        nn.BatchNorm3d(channel[1]),
        nn.LeakyReLU(0.2, inplace=True)
    )
class R3Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 512
        self.out_dim = 4
        self.conv1 = conv3d((1,64))
        self.conv2 = conv3d((64,128))
        self.conv3 = conv3d((128,256))
        self.conv4 = conv3d((256,512))
        self.out = nn.Sequential(
            nn.Linear(512 * self.out_dim * self.out_dim * self.out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply linear + sigmoid
        x = x.view(-1, self.out_channels * self.out_dim * self.out_dim * self.out_dim)
        x = self.out(x)
        return x