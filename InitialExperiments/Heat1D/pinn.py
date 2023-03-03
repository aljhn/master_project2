import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import functorch


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def heat(t, x):
    return torch.sin(np.pi * x) * torch.exp(-(np.pi**2.0) * t)


model = nn.Sequential(
    nn.Linear(2, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 1)
)

batch_size = 100

T0 = 0.0
T1 = 0.2
X0 = 0.0
X1 = 1.0

t_i = torch.ones((batch_size // 2, 1)) * T0
x_i = torch.rand((batch_size // 2, 1)) * (X1 - X0) + X0
u_i = heat(t_i, x_i)
tx_i = torch.cat((t_i, x_i), dim=1)

t_b = torch.rand((batch_size // 2, 1)) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((batch_size // 4, 1)) * X0, torch.ones((batch_size // 4, 1)) * X1), dim=0)
tx_b = torch.cat((t_b, x_b), dim=1)
u_b = heat(t_b, x_b)

tx_ib = torch.cat((tx_i, tx_b), dim=0)
u_ib = torch.cat((u_i, u_b), dim=0)

n_pinn = 1000
t_pinn = torch.rand(n_pinn) * (T1 - T0) + T0
x_pinn = torch.rand(n_pinn) * (X1 - X0) + X0
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
    plt.rcParams["font.family"] = "Times New Roman"

    n = 100
    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    uu = model(ttxx)
    un = torch.reshape(uu, tt.shape)
    uu_true = heat(tt, xx)

    print("True difference:", torch.mean((un - uu_true)**2))

    plt.figure()
    plt.pcolormesh(t, x, un, vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("heat1d.pdf")
    plt.show()

    plt.figure()
    plt.pcolormesh(t, x, uu_true, vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("heat1d_true.pdf")
    plt.show()
