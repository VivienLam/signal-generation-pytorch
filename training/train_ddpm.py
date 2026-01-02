import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models.diffusion import Diffusion1D
from models.noise_predictor import NoisePredictor

signals = torch.tensor(np.load("data/signals.npy"), dtype=torch.float32)

dataset = TensorDataset(signals)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


diffusion = Diffusion1D(timesteps=100)
model = NoisePredictor(signal_dim=signals.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

n_epochs = 50

for epoch in range(n_epochs):
    total_loss = 0

    for (x0,) in loader:
        t = torch.randint(0, diffusion.timesteps, (x0.shape[0],))
        xt, noise = diffusion.add_noise(x0, t)

        noise_pred = model(xt, t)
        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")


model.eval()
with torch.no_grad():
    generated = diffusion.sample(model, signals.shape[1])

t = np.linspace(0, 1, signals.shape[1])

for i in range(generated.shape[0]):
    plt.plot(t, generated[i].numpy())
    plt.title("Signal généré (diffusion)")
    plt.show()

torch.save(model.state_dict(), "checkpoints/ddpm.pt")