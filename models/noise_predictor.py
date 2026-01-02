import torch
import torch.nn as nn

class NoisePredictor(nn.Module):
    def __init__(self, signal_dim=100):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(signal_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, signal_dim)
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(1) / 100
        x = torch.cat([x, t], dim=1)
        return self.net(x)
