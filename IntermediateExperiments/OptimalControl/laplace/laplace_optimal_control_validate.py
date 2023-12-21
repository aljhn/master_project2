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


def boundary_condition(x):
    return torch.sin(np.pi * x)


def q_d(x):
    return torch.cos(np.pi * x)


X0 = 0.0
X1 = 1.0
Y0 = 0.0
Y1 = 1.0

n_data = 100

x_boundary = torch.rand(n_data, device=device) * (X1 - X0) + X0
y_boundary = torch.ones(n_data, device=device) * Y0
u_boundary = boundary_condition(x_boundary).unsqueeze(1)
xy_boundary = torch.stack((x_boundary, y_boundary), dim=1)

x_control = torch.linspace(X0, X1, n_data, device=device)
y_control = torch.ones(n_data, device=device) * Y1
xy_control = torch.stack((x_control, y_control), dim=1)
u_control = torch.tensor(
    np.loadtxt("laplace_control"), device=device, dtype=torch.float32
).unsqueeze(1)

x_periodic_boundary0 = torch.ones(n_data, device=device) * X0
y_periodic_boundary0 = torch.rand(n_data, device=device) * (Y1 - Y0) + Y0
xy_periodic_boundary0 = torch.stack((x_periodic_boundary0, y_periodic_boundary0), dim=1)

x_periodic_boundary1 = torch.ones(n_data, device=device) * X1
y_periodic_boundary1 = torch.rand(n_data, device=device) * (Y1 - Y0) + Y0
xy_periodic_boundary1 = torch.stack((x_periodic_boundary1, y_periodic_boundary1), dim=1)

xy_periodic_boundary = torch.cat((xy_periodic_boundary0, xy_periodic_boundary1), dim=0)

xy_data = torch.cat((xy_boundary, xy_control, xy_periodic_boundary), dim=0)

n_pinn = 10000
x_pinn = torch.rand(n_pinn, device=device) * (X1 - X0) + X0
y_pinn = torch.rand(n_pinn, device=device) * (Y1 - Y0) + Y0
xy_pinn = torch.stack((x_pinn, y_pinn), dim=1)

criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(
    (*model_u.parameters(),),
    lr=1,
    max_iter=10000,
    line_search_fn="strong_wolfe",
)
# optimizer = torch.optim.LBFGS(model_u.parameters(), lr=1, max_iter=10000, line_search_fn="strong_wolfe")

model_u_jacobian = torch.vmap(torch.func.grad(lambda x: model_u(x).squeeze()))
model_u_hessian = torch.vmap(torch.func.hessian(lambda x: model_u(x).squeeze()))

max_epochs = 5000
epoch = 0

beta_b = 1.0
beta_f = 1.0
beta_j = 100.0

n_j = 41
h_j = 1.0 / n_j
x_j = torch.linspace(X0, X1, n_j, device=device)
u_j = q_d(x_j)
y_j = torch.ones(n_j, device=device) * Y1
xy_j = torch.stack((x_j, y_j), dim=1)


def trapezoid(f, h):
    return h * (torch.sum(f[1:-1]) + (f[0] + f[-1]) / 2.0)


losses = np.zeros((max_epochs, 3))


def closure():
    global epoch
    epoch += 1

    if epoch > max_epochs:
        raise KeyboardInterrupt

    optimizer.zero_grad()

    boundary_loss = 0

    u_data = model_u(xy_data)

    u_boundary_pred = u_data[:n_data, :]
    boundary_loss += criterion(u_boundary_pred, u_boundary)

    u_control_pred = u_data[n_data : 2 * n_data, :]
    boundary_loss += criterion(u_control_pred, u_control)

    u_periodic_boundary0 = u_data[2 * n_data : 3 * n_data, :]
    u_periodic_boundary1 = u_data[3 * n_data :, :]
    boundary_loss += criterion(u_periodic_boundary0, u_periodic_boundary1)

    u_periodic_boundary_jacobian = model_u_jacobian(xy_periodic_boundary)
    u_periodic_boundary_jacobian0 = u_periodic_boundary_jacobian[:n_data, 0]
    u_periodic_boundary_jacobian1 = u_periodic_boundary_jacobian[n_data:, 0]
    boundary_loss += criterion(
        u_periodic_boundary_jacobian0, u_periodic_boundary_jacobian1
    )

    u_hessian = model_u_hessian(xy_pinn)
    u_laplacian = torch.vmap(torch.trace)(u_hessian)
    f = u_laplacian
    physics_loss = torch.mean(f**2)

    u_j_jacobian = model_u_jacobian(xy_j)
    u_j_dy = u_j_jacobian[:, 1]
    j = (u_j_dy - u_j) ** 2.0
    cost = trapezoid(j, h_j)

    loss = beta_b * boundary_loss + beta_f * physics_loss
    loss.backward()

    losses[epoch - 1, 0] = boundary_loss.item()
    losses[epoch - 1, 1] = physics_loss.item()
    losses[epoch - 1, 2] = cost.item()

    print(
        f"Epoch: {epoch:5d}, B: {boundary_loss.item():.8f}, F: {physics_loss.item():.8f}, C: {cost.item():.8f}"
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
    x = torch.linspace(X0, X1, n, device=device)
    y = torch.linspace(Y0, Y1, n, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xxyy = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    uu = model_u(xxyy)
    un = torch.reshape(uu, xx.shape)
    # uu_true = solution_u(xx, yy)
    # print("True difference:", torch.mean((un - uu_true)**2))

    x = x.cpu()
    y = y.cpu()
    un = un.cpu()

    # plt.figure()
    # plt.pcolormesh(x, y, uu_true, cmap="rainbow")
    # plt.colorbar()
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$y$")
    # plt.tight_layout()
    # plt.savefig("laplace_optimal_control_true.pdf")
    # plt.show()

    plt.figure()
    plt.pcolormesh(x, y, un, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.tight_layout()
    plt.savefig("laplace_optimal_control_validate.pdf")
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, max_epochs + 1), losses[:, 0])
    plt.plot(np.arange(1, max_epochs + 1), losses[:, 1])
    plt.plot(np.arange(1, max_epochs + 1), losses[:, 2])
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Loss")
    plt.legend(["Boundary", "Physics", "Cost"])
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("laplace_optimal_control_losses_validate.pdf")
    plt.show()
