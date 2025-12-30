import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=20, signal_dim=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, signal_dim)
        )

    def forward(self, z):
        return self.net(z)

