import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models.autoencoder import AutoEncoder

signals = np.load("data/signals.npy")
signals = torch.tensor(signals, dtype=torch.float32)

dataset = TensorDataset(signals)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

signal_dim = signals.shape[1]
latent_dim = 8

model = AutoEncoder(signal_dim, latent_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

n_epochs = 100

for epoch in range(n_epochs):
    total_loss = 0.0

    for (x,) in loader:
        x_hat, z = model(x)
        loss = loss_fn(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {total_loss/len(loader):.4f}")

model.eval()

with torch.no_grad():
    x = signals[:5]
    x_hat, z = model(x)

t = np.linspace(0, 1, signal_dim)

for i in range(5):
    plt.figure()
    plt.plot(t, x[i].numpy(), label="Original")
    plt.plot(t, x_hat[i].numpy(), label="Reconstruit")
    plt.legend()
    plt.title(f"Signal {i}")
    plt.show()

z_all = []

with torch.no_grad():
    for (x,) in loader:
        _, z = model(x)
        z_all.append(z)

z_all = torch.cat(z_all).numpy()
torch.save(model.state_dict(), "checkpoints/ae.pt")
