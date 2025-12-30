import torch

x = torch.randn(5)
w = torch.randn(5, requires_grad=True)

y = (x * w).sum()
y.backward()

print("Gradient :", w.grad)
