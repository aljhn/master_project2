import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


model_u = nn.Sequential(
    nn.Linear(2, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 1)
)


model_c = nn.Sequential(
    nn.Linear(2, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 1)
)


def initial_condition(x):
    return torch.sin(np.pi * x)


def boundary_condition(t):
    return torch.zeros_like(t)


T0 = 0.0
T1 = 0.2
X0 = 0.0
X1 = 1.0

n_data = 100

t_i = torch.ones((n_data // 2, 1)) * T0
x_i = torch.rand((n_data // 2, 1)) * (X1 - X0) + X0
u_i = initial_condition(x_i)
tx_i = torch.cat((t_i, x_i), dim=1)

t_b = torch.rand((n_data // 2, 1)) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((n_data // 4, 1)) * X0, torch.ones((n_data // 4, 1)) * X1), dim=0)
u_b = boundary_condition(t_b)
tx_b = torch.cat((t_b, x_b), dim=1)

tx_ib = torch.cat((tx_i, tx_b), dim=0)
u_ib = torch.cat((u_i, u_b), dim=0)


n_pinn = 10000
t_pinn = torch.rand((n_pinn, 1)) * (T1 - T0) + T0
x_pinn = torch.rand((n_pinn, 1)) * (X1 - X0) + X0
tx_pinn = torch.cat((t_pinn, x_pinn), dim=1)


n_cost = 1000
t_j = torch.ones((n_cost, 1)) * T1
x_j = torch.rand((n_cost, 1)) * (X1 - X0) + X0
u_j = 0.5 * torch.sin(x_j)
tx_j = torch.cat((t_j, x_j), dim=1)


criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS((*model_u.parameters(), *model_c.parameters()), lr=1, max_iter=10000, line_search_fn="strong_wolfe")
# optimizer = torch.optim.LBFGS(model_u.parameters(), lr=1, max_iter=10000, line_search_fn="strong_wolfe")

model_u_jacobian = torch.vmap(torch.func.grad(lambda x: model_u(x).squeeze()))
model_u_hessian = torch.vmap(torch.func.hessian(lambda x: model_u(x).squeeze()))

max_epochs = 10000
epoch = 0

beta_b = 1.0
beta_f = 1.0
beta_j = 1.0


def closure():
    global epoch
    epoch += 1

    if epoch > max_epochs:
        raise KeyboardInterrupt

    optimizer.zero_grad()

    ib_loss = 0

    u_pred = model_u(tx_ib)
    ib_loss += criterion(u_pred, u_ib)

    u_t = model_u_jacobian(tx_pinn)[:, 0]
    u_xx = model_u_hessian(tx_pinn)[:, 1, 1]
    c = model_c(tx_pinn)[:, 0]
    f = u_t - u_xx - c
    physics_loss = torch.mean(f**2)

    u_j_pred = model_u(tx_j)
    cost = criterion(u_j_pred, u_j)

    loss = beta_b * ib_loss + beta_f * physics_loss + beta_j * cost
    loss.backward()

    print(f"Epoch: {epoch:5d}, IB: {ib_loss.item():.8f}, F: {physics_loss.item():.8f}, C: {cost.item():.8f}")

    return loss
    

while True:
    try:
        optimizer.step(closure)
    except KeyboardInterrupt:
        break


with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    n = 100
    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    uu = model_u(ttxx)
    un = torch.reshape(uu, tt.shape)

    plt.figure()
    # plt.pcolormesh(t, x, un, cmap="rainbow")
    plt.pcolormesh(t, x, un, cmap="rainbow", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control.pdf")
    plt.show()

    plt.figure()
    cc = model_c(ttxx)
    cn = torch.reshape(cc, tt.shape)
    plt.pcolormesh(t, x, cn, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control_control.pdf")
    plt.show()

    plt.figure()
    t = torch.ones(n) * T1
    tx = torch.stack((t, x), dim=1)
    u = model_u(tx)
    plt.plot(x, u)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control_slice.pdf")
    plt.show()
    