import torch
import torch.nn as nn
from attention import R3DAttention

class R3DEncoder(nn.Module):
    def __init__(self, inChannel =1, imSize = (192,256), latent_dim=1024):
        super().__init__()
        self.conv1_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannel, self.conv1_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.attention1 = R3DAttention(self.conv1_channel, 4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.conv1_channel, 2*self.conv1_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(4),
            nn.ReLU()
        ) 
        self.attention2 = R3DAttention(2*self.conv1_channel, 8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*self.conv1_channel, 4*self.conv1_channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(4),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(4*self.conv1_channel*6*8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() 
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        # apply encoder network to input image
        x = self.conv1(x)
        x = x + self.attention1(x)
        x = self.conv2(x)
        x = x + self.attention2(x)
        x = self.conv3(x)
        # x = x.reshape((4*self.conv1_channel,6, 32, 8, 32)).permute((2, 4, 0, 1, 3)).reshape((1024, -1))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_logvar(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z