import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Boundary2: Dirichlet
# Boundary1: Neumann


model_u = nn.Sequential(
    nn.Linear(2, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 1),
)
model_u.to(device)


model_c = nn.Sequential(
    nn.Linear(1, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 1),
)
model_c.to(device)


def initial_condition(x):
    return torch.sin(np.pi * x)


T0 = 0.0
T1 = 1.0
X0 = 0.0
X1 = 1.0

n_data = 1000

t_i = torch.ones((n_data // 2, 1), device=device) * T0
x_i = torch.rand((n_data // 2, 1), device=device) * (X1 - X0) + X0
u_i = initial_condition(x_i)
tx_i = torch.cat((t_i, x_i), dim=1)


t_b = torch.rand((n_data // 2, 1), device=device) * (T1 - T0) + T0
x_b = torch.ones((n_data // 2, 1), device=device) * X0
tx_b = torch.cat((t_b, x_b), dim=1)


n_pinn = 10000
t_pinn = torch.rand((n_pinn, 1), device=device) * (T1 - T0) + T0
x_pinn = torch.rand((n_pinn, 1), device=device) * (X1 - X0) + X0
tx_pinn = torch.cat((t_pinn, x_pinn), dim=1)


n_cost = 1000
t_j = torch.ones((n_cost, 1), device=device) * T1
x_j = torch.rand((n_cost, 1), device=device) * (X1 - X0) + X0
u_j = 0.5 * torch.ones((n_cost, 1), device=device)
tx_j = torch.cat((t_j, x_j), dim=1)


criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(
    (*model_u.parameters(), *model_c.parameters()),
    lr=1,
    max_iter=10000,
    line_search_fn="strong_wolfe",
)
# optimizer = torch.optim.LBFGS(model_u.parameters(), lr=1, max_iter=10000, line_search_fn="strong_wolfe")

model_u_jacobian = torch.vmap(torch.func.grad(lambda x: model_u(x).squeeze()))
model_u_hessian = torch.vmap(torch.func.hessian(lambda x: model_u(x).squeeze()))

max_epochs = 2000
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

    u_pred = model_u(tx_i)
    ib_loss += criterion(u_pred, u_i)

    u_b_jacobian = model_u_jacobian(tx_b)[:, 1]
    c = model_c(t_b)[:, 0]
    ib_loss += criterion(u_b_jacobian, c)

    u_t = model_u_jacobian(tx_pinn)[:, 0]
    u_xx = model_u_hessian(tx_pinn)[:, 1, 1]
    f = u_t - u_xx
    physics_loss = torch.mean(f**2)

    u_j_pred = model_u(tx_j)
    cost = criterion(u_j_pred, u_j)

    loss = beta_b * ib_loss + beta_f * physics_loss + beta_j * cost
    loss.backward()

    print(
        f"Epoch: {epoch:5d}, IB: {ib_loss.item():.8f}, F: {physics_loss.item():.8f}, C: {cost.item():.8f}"
    )

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
    t = torch.linspace(T0, T1, n, device=device)
    x = torch.linspace(X0, X1, n, device=device)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    uu = model_u(ttxx)
    un = torch.reshape(uu, tt.shape)

    plt.figure()
    plt.pcolormesh(t.cpu(), x.cpu(), un.cpu(), cmap="rainbow", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control_boundary1.pdf")
    plt.show()

    cc = model_c(t.unsqueeze(1))

    plt.figure()
    plt.plot(t.cpu(), cc.cpu()[:, 0])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$c$")
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control_boundary1_control.pdf")
    plt.show()

    t1 = torch.ones(n, device=device) * T0
    t2 = torch.ones(n, device=device) * (T0 + T1) / 2.0
    t3 = torch.ones(n, device=device) * T1
    tx1 = torch.stack((t1, x), dim=1)
    tx2 = torch.stack((t2, x), dim=1)
    tx3 = torch.stack((t3, x), dim=1)
    tx = torch.cat((tx1, tx2, tx3), dim=0)
    u = model_u(tx)
    u1 = u[:n, 0]
    u2 = u[n : 2 * n, 0]
    u3 = u[2 * n :, 0]

    plt.figure()
    plt.plot(x.cpu(), u1.cpu())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control_boundary1_slice1.pdf")
    plt.show()

    plt.figure()
    plt.plot(x.cpu(), u2.cpu())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control_boundary1_slice2.pdf")
    plt.show()

    plt.figure()
    plt.plot(x.cpu(), u3.cpu())
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig("heat1d_optimal_control_boundary1_slice3.pdf")
    plt.show()
