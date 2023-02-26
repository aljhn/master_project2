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
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 1)
)


def initial_condition(x):
    return -torch.sin(np.pi * x)


def boundary_condition(t):
    return torch.zeros_like(t)



T0 = 0.0
T1 = 1.0
X0 = -1.0
X1 = 1.0

nu = 0.01 / np.pi

n_ib = 100

t_i = torch.ones((n_ib // 2, 1)) * T0
x_i = torch.rand((n_ib // 2, 1)) * (X1 - X0) + X0
tx_i = torch.cat((t_i, x_i), dim=1)
u_i = initial_condition(x_i)

t_b = torch.rand((n_ib // 2, 1)) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((n_ib // 4, 1)) * X0, torch.ones((n_ib // 4, 1)) * X1), dim=0)
tx_b = torch.cat((t_b, x_b), dim=1)
u_b = boundary_condition(t_b)

tx_ib = torch.cat((tx_i, tx_b), dim=0)
u_ib = torch.cat((u_i, u_b), dim=0)

n_pinn = 10000
# t_pinn = torch.linspace(T0, T1, n_pinn)
# x_pinn = torch.linspace(X0, X1, n_pinn)
t_pinn = torch.rand(n_pinn) * (T1 - T0) + T0
x_pinn = torch.rand(n_pinn) * (X1 - X0) + X0
tx_pinn = torch.stack((t_pinn, x_pinn), dim=1)

# tx_pinn.requires_grad = True

criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=10000, line_search_fn="strong_wolfe")

model_jacobian = functorch.vmap(functorch.grad(lambda x: model(x).squeeze()))
model_hessian = functorch.vmap(functorch.hessian(lambda x: model(x).squeeze()))

max_epochs = 1000
epoch = 0


def closure():
    global epoch
    epoch += 1

    if epoch > max_epochs:
        raise KeyboardInterrupt

    optimizer.zero_grad()

    u_pred = model(tx_ib)
    loss = criterion(u_pred, u_ib)

    u = model(tx_pinn)[:, 0]

    u_jacobian = model_jacobian(tx_pinn)
    u_t = u_jacobian[:, 0]
    u_x = u_jacobian[:, 1]
    u_xx = model_hessian(tx_pinn)[:, 1, 1]

    # u_grad = torch.autograd.grad(u, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    # u_t = u_grad[:, 0]
    # u_x = u_grad[:, 1]
    # u_x_grad = torch.autograd.grad(u_x, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    # u_xx = u_x_grad[:, 1]

    f = u_t + u * u_x - nu * u_xx
    loss += torch.mean(f**2)

    loss.backward()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")

    return loss
    

while True:
    try:
        optimizer.step(closure)
    except KeyboardInterrupt:
        break


with torch.no_grad():
    n = 100
    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    uu = model(ttxx)
    un = torch.reshape(uu, tt.shape)
    plt.figure()
    plt.pcolormesh(t, x, un, vmin=-1.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.title(r"$u(t, x)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.savefig("burger.pdf")
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
    plt.savefig("burger_slice.pdf")
    plt.show()
