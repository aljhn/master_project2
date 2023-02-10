import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import random
seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


model = nn.Sequential(
    nn.Linear(2, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 1)
)

t = torch.zeros((100, 1))
x = torch.linspace(0, 1, 100).unsqueeze(1)
u = torch.sin(3.14 * x)
tx = torch.cat((t, x), dim=1)

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
    plt.figure()
    plt.scatter(x, u)
    uu = model(tx)
    plt.scatter(x, uu)
    plt.show()
