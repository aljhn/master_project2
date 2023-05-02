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
    nn.Linear(1, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 1)
)


T0 = 0.0
T1 = 5.0
X0 = 0.0
X1 = 4.0

nu = 0.01


def solution(t, x):
    return 2.0 * nu * np.pi * torch.exp(-(np.pi**2.0) * nu * (t - 5.0)) * torch.sin(np.pi * x) \
        / (2.0 + torch.exp(-(np.pi**2.0) * nu * (t - 5.0)) * torch.cos(np.pi * x))


n_data = 400

t_i = torch.ones((n_data // 2, 1)) * T0
x_i = torch.rand((n_data // 2, 1)) * (X1 - X0) + X0
tx_i = torch.cat((t_i, x_i), dim=1)

t_b = torch.rand((n_data // 2, 1)) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((n_data // 4, 1)) * X0, torch.ones((n_data // 4, 1)) * X1), dim=0)
tx_b = torch.cat((t_b, x_b), dim=1)
u_b = torch.zeros_like(t_b)

tx_ib = torch.cat((tx_i, tx_b), dim=0)


n_pinn = 20000
t_pinn = torch.rand((n_pinn, 1)) * (T1 - T0) + T0
x_pinn = torch.rand((n_pinn, 1)) * (X1 - X0) + X0
tx_pinn = torch.cat((t_pinn, x_pinn), dim=1)


criterion = nn.MSELoss()
# optimizer = torch.optim.LBFGS((*model_u.parameters(), *model_c.parameters()), lr=1, max_iter=10000, line_search_fn="strong_wolfe")
optimizer = torch.optim.Adam((*model_u.parameters(), *model_c.parameters()), lr=1e-3)

model_u_jacobian = torch.vmap(torch.func.grad(lambda x: model_u(x).squeeze()))
model_u_hessian = torch.vmap(torch.func.hessian(lambda x: model_u(x).squeeze()))

max_epochs = 20000
epoch = 0

beta_i = 1.0
beta_b = 1.0
beta_f = 1.0
beta_j = 1.0


n_j = 41
h_j = 1.0 / n_j
x_j = torch.linspace(X0, X1, n_j)
u_j = solution(torch.tensor([T1]), x_j)
t_j = torch.ones(n_j) * T1
tx_j = torch.stack((t_j, x_j), dim=1)


def trapezoid(f, h):
    return h * (torch.sum(f[1:-1]) + (f[0] + f[-1]) / 2.0)


i_losses = []
b_losses = []
f_losses = []
j_losses = []


def closure():
    global epoch
    epoch += 1

    if epoch > max_epochs:
        raise KeyboardInterrupt

    optimizer.zero_grad()

    u_ib_pred = model_u(tx_ib)

    u_i = model_c(x_i)
    # u_i = solution(torch.tensor([T0]), x_i)
    u_i_pred = u_ib_pred[:n_data // 2, :]
    initial_loss = criterion(u_i_pred, u_i)

    u_b_pred = u_ib_pred[n_data // 2:, :]
    boundary_loss = criterion(u_b_pred, u_b)

    u = model_u(tx_pinn)[:, 0]
    u_jacobian = model_u_jacobian(tx_pinn)
    u_t = u_jacobian[:, 0]
    u_x = u_jacobian[:, 1]
    u_xx = model_u_hessian(tx_pinn)[:, 1, 1]

    f = u_t + u * u_x - nu * u_xx
    physics_loss = torch.mean(f**2)

    u_j_pred = model_u(tx_j)[:, 0]
    j = 0.5 * ((u_j_pred - u_j)**2.0)
    cost = trapezoid(j, h_j)

    loss = beta_i * initial_loss + beta_b * boundary_loss + beta_f * physics_loss + beta_j * cost
    loss.backward()

    print(f"Epoch: {epoch:5d}, I: {initial_loss.item():.8f}, B: {boundary_loss.item():.8f}, F: {physics_loss.item():.8f}, C: {cost.item():.8f}")

    i_losses.append(initial_loss.item())
    b_losses.append(boundary_loss.item())
    f_losses.append(physics_loss.item())
    j_losses.append(cost.item())

    return loss
    

while True:
    try:
        optimizer.step(closure)
        if epoch == 18000:
            optimizer = torch.optim.LBFGS((*model_u.parameters(), *model_c.parameters()), lr=1, max_iter=10000, line_search_fn="strong_wolfe")
    except KeyboardInterrupt:
        break


with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    plt.figure()
    plt.plot(np.arange(1, len(i_losses) + 1), i_losses)
    plt.plot(np.arange(1, len(b_losses) + 1), b_losses)
    plt.plot(np.arange(1, len(f_losses) + 1), f_losses)
    plt.plot(np.arange(1, len(j_losses) + 1), j_losses)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Loss")
    plt.legend(["Initial", "Boundary", "Physics", "Cost"])
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("burger_control_losses.pdf")
    plt.show()

    n = 100
    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")

    un = solution(tt, xx)
    plt.figure()
    plt.pcolormesh(t, x, un, vmin=-0.1, vmax=0.1, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("burger_control_true.pdf")
    plt.show()

    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    uu = model_u(ttxx)
    unn = torch.reshape(uu, tt.shape)
    plt.figure()
    plt.pcolormesh(t, x, unn, vmin=-0.1, vmax=0.1, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("burger_control.pdf")
    plt.show()

    plt.figure()
    plt.plot(x, un[:, 0])
    plt.plot(x, unn[:, 0])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.legend(["True Solution", "Learned Solution"])
    plt.ylim([-0.1, 0.1])
    plt.tight_layout()
    plt.savefig("burger_control_slice0.pdf")
    plt.show()

    plt.figure()
    plt.plot(x, un[:, -1])
    plt.plot(x, unn[:, -1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.legend(["True Solution", "Learned Solution"])
    plt.ylim([-0.1, 0.1])
    plt.tight_layout()
    plt.savefig("burger_control_slice1.pdf")
    plt.show()
