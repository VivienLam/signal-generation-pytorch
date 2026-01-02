import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models.vae import VAE

def vae_loss(x_hat, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


signals = np.load("data/signals.npy")
signals = torch.tensor(signals, dtype=torch.float32)

dataset = TensorDataset(signals)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = VAE(signal_dim=signals.shape[1], latent_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100

for epoch in range(n_epochs):
    total_loss = 0

    for (x,) in loader:
        x_hat, mu, logvar = model(x)
        loss = vae_loss(x_hat, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")

model.eval()

with torch.no_grad():
    z = torch.randn(5, 8)
    generated = model.decoder(z)

t = np.linspace(0, 1, signals.shape[1])

for i in range(5):
    plt.plot(t, generated[i].numpy())
    plt.title("Signal généré")
    plt.show()

torch.save(model.state_dict(), "checkpoints/vae.pt")
