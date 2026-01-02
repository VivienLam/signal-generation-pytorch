import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, signal_dim=100, latent_dim=8):
        super().__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(signal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, signal_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
