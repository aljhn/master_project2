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
T1 = 1.0
X0 = -1.0
X1 = 1.0


def initial_condition(x):
    return -torch.sin(3.14 * x)

def boundary_condition(t):
    return torch.zeros_like(t)


t_i = torch.ones((batch_size // 2, 1)) * T0
x_i = torch.rand((batch_size // 2, 1)) * (X1 - X0) + X0
tx_i = torch.cat((t_i, x_i), dim=1)
u_i = initial_condition(x_i)

t_b = torch.rand((batch_size // 2, 1)) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((batch_size // 4, 1)) * X0, torch.ones((batch_size // 4, 1)) * X1), dim=0)
tx_b = torch.cat((t_b, x_b), dim=1)
u_b = boundary_condition(t_b)

tx_ib = torch.cat((tx_i, tx_b), dim=0)
u_ib = torch.cat((u_i, u_b), dim=0)

n_pinn = 1000
t_pinn = torch.linspace(T0, T1, n_pinn)
x_pinn = torch.linspace(X0, X1, n_pinn)
tx_pinn = torch.stack((t_pinn, x_pinn), dim=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

model_jacobian = functorch.vmap(functorch.grad(lambda x: model(x).squeeze()))
model_hessian = functorch.vmap(functorch.hessian(lambda x: model(x).squeeze()))

epochs = 1000
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()
        u_pred = model(tx_ib)
        loss = criterion(u_pred, u_ib)

        u = model(tx_pinn)[:, 0]
        u_jacobian = model_jacobian(tx_pinn)
        u_t = u_jacobian[:, 0]
        u_x = u_jacobian[:, 1]
        u_xx = model_hessian(tx_pinn)[:, 1, 1]
        f = u_t + u * u_x - 0.01 / 3.14 * u_xx
        loss += 0.1 * torch.mean(f**2)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    except KeyboardInterrupt:
        break


with torch.no_grad():
    n = 60
    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt = torch.zeros((n * n, 1))
    xx = torch.zeros((n * n, 1))
    for i in range(n):
        for j in range(n):
            tt[i * n + j, 0] = t[i]
            xx[i * n + j, 0] = x[j]
    ttxx = torch.cat((tt, xx), dim=1)
    uu = model(ttxx)
    un = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            un[j, i] = uu[i * n + j]
    plt.figure()
    plt.pcolormesh(t, x, un, vmin=-1.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.title(r"$u(t, x)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.savefig("burger_naive.pdf")
    plt.show()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(f"t={T0:.2f}")
    plt.plot(x, un[:, 0])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.subplot(2, 2, 2)
    plt.title(f"t={(T1 - T0) / 3:.2f}")
    plt.plot(x, un[:, n // 3 - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.subplot(2, 2, 3)
    plt.title(f"t={(T1 - T0) * 2 / 3:.2f}")
    plt.plot(x, un[:, n * 2 // 3 - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.subplot(2, 2, 4)
    plt.title(f"t={T1:.2f}")
    plt.plot(x, un[:, n - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice_naive.pdf")
    plt.show()
