import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import functorch

import random
seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def heat(t, x):
    return torch.sin(3.14 * x) * torch.exp(-(3.14**2.0) * t)


model = nn.Sequential(
    nn.Linear(2, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 1)
)

batch_size = 400

T0 = 0.0
T1 = 0.2
X0 = 0.0
X1 = 1.0

t_i = torch.ones((batch_size // 2, 1)) * T0
x_i = torch.rand((batch_size // 2, 1)) * (X1 - X0) + X0
u_i = heat(t_i, x_i)
tx_i = torch.cat((t_i, x_i), dim=1)

t_b = torch.rand((batch_size // 2, 1)) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((batch_size // 4, 1)), torch.ones(batch_size // 4, 1)), dim=0)
tx_b = torch.cat((t_b, x_b), dim=1)
u_b = heat(t_b, x_b)

tx_ib = torch.cat((tx_i, tx_b), dim=0)
u_ib = torch.cat((u_i, u_b), dim=0)

n_pinn = 1000
t_pinn = torch.linspace(0.0, 0.2, n_pinn)
x_pinn = torch.linspace(0.0, 1.0, n_pinn)
tx_pinn = torch.stack((t_pinn, x_pinn), dim=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model_jacobian = functorch.vmap(functorch.grad(lambda x: model(x).squeeze()))
model_hessian = functorch.vmap(functorch.hessian(lambda x: model(x).squeeze()))

epochs = 1000
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()
        u_pred = model(tx_ib)
        loss = criterion(u_pred, u_ib)

        u_t = model_jacobian(tx_pinn)[:, 0]
        u_xx = model_hessian(tx_pinn)[:, 1, 1]
        f = u_t - u_xx
        loss += 0.1 * torch.mean(f**2)

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
    plt.pcolormesh(t, x, un, vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.show()
