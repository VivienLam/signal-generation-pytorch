import torch
import numpy as np
import matplotlib.pyplot as plt

from models.autoencoder import AutoEncoder
from models.vae import VAE
from models.diffusion import Diffusion1D
from models.noise_predictor import NoisePredictor

# -----------------------
# Paramètres
# -----------------------
signal_dim = 100
latent_dim = 8
n_samples = 3

# -----------------------
# Chargement des données
# -----------------------
signals = torch.tensor(
    np.load("data/signals.npy"),
    dtype=torch.float32
)

x_real = signals[:n_samples]

t_axis = np.linspace(0, 1, signal_dim)

# -----------------------
# Autoencodeur
# -----------------------
ae = AutoEncoder(signal_dim, latent_dim)
ae.load_state_dict(torch.load("checkpoints/ae.pt"))
ae.eval()

with torch.no_grad():
    x_ae, _ = ae(x_real)

# -----------------------
# VAE
# -----------------------
vae = VAE(signal_dim, latent_dim)
vae.load_state_dict(torch.load("checkpoints/vae.pt"))
vae.eval()

with torch.no_grad():
    z = torch.randn(n_samples, latent_dim)
    x_vae = vae.decoder(z)

# -----------------------
# Diffusion
# -----------------------
diffusion = Diffusion1D(timesteps=100)
ddpm = NoisePredictor(signal_dim)
ddpm.load_state_dict(torch.load("checkpoints/ddpm.pt"))
ddpm.eval()

with torch.no_grad():
    x_diff = diffusion.sample(ddpm, signal_dim, n_samples)

# -----------------------
# Visualisation
# -----------------------
for i in range(n_samples):
    plt.figure(figsize=(10, 6))

    plt.plot(t_axis, x_real[i].numpy(), label="Original", linewidth=2)
    plt.plot(t_axis, x_ae[i].numpy(), label="AE reconstruction")
    plt.plot(t_axis, x_vae[i].numpy(), label="VAE generation")
    plt.plot(t_axis, x_diff[i].numpy(), label="Diffusion generation")

    plt.legend()
    plt.title(f"Comparison of generative models – sample {i}")
    plt.xlabel("Time")
    plt.ylabel("Signal value")

    plt.show()
