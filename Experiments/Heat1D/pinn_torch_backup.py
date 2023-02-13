import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def heat(t, x):
    return torch.sin(3.14 * x) * torch.exp(-(3.14**2.0) * t)


model = nn.Sequential(
    nn.Linear(2, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 1)
)

batch_size = 400
t = torch.rand((batch_size, 1)) * 0.2
x = torch.rand((batch_size, 1)) * 1.0
tx = torch.cat((t, x), dim=1)
u = heat(t, x)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 1000
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()
        u_pred = model(tx)
        loss = criterion(u_pred, u)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    except KeyboardInterrupt:
        break


with torch.no_grad():
    n = 50
    t = torch.linspace(0.0, 0.2, n)
    x = torch.linspace(0.0, 1.0, n)
    tt = torch.zeros((n * n, 1))
    xx = torch.zeros((n * n, 1))
    for i in range(n):
        for j in range(n):
            tt[i * n + j, 0] = t[i]
            xx[i * n + j, 0] = x[j]
    ttxx = torch.cat((tt, xx), dim=1)
    uu = model(ttxx)
    uu_true = heat(tt, xx)
    print("True difference:", torch.mean((uu - uu_true)**2))
    un = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            un[j, i] = uu[i * n + j]
            # un[j, i] = uu_true[i * n + j]
    plt.figure()
    plt.pcolormesh(t, x, un)
    plt.colorbar()
    plt.show()
