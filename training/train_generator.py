import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.mlp_generator import Generator

# Charger les signaux
signals = np.load("data/signals.npy")

signals = torch.tensor(signals, dtype=torch.float32)

dataset = TensorDataset(signals)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

latent_dim = 20
signal_dim = signals.shape[1]

model = Generator(latent_dim, signal_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

n_epochs = 50

for epoch in range(n_epochs):
    total_loss = 0.0

    for (real_signals,) in dataloader:
        batch_size = real_signals.size(0)

        # Bruit aléatoire
        z = torch.randn(batch_size, latent_dim)

        # Génération
        generated = model(z)

        # Loss : rapprocher signaux générés des vrais
        loss = loss_fn(generated, real_signals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")


import matplotlib.pyplot as plt

model.eval()

with torch.no_grad():
    z = torch.randn(5, latent_dim)
    generated = model(z).numpy()

t = np.linspace(0, 1, signal_dim)

for i in range(5):
    plt.plot(t, generated[i])

plt.title("Signaux générés par le modèle")
plt.show()
