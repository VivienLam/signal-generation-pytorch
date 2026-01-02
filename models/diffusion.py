import torch
import torch.nn as nn
import numpy as np

class Diffusion1D:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        noisy_x = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_x, noise

    def sample(self, model, signal_dim, n_samples=5):
        x = torch.randn(n_samples, signal_dim)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((n_samples,), t)
            noise_pred = model(x, t_tensor)

            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
            )

        return x