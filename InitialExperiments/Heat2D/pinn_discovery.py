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


def heat(t, x, y, k=1.0):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.exp(-((k * np.pi)**2.0) * 2 * t)


model = nn.Sequential(
    nn.Linear(3, 20),
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

T0 = 0.0
T1 = 0.2
X0 = 0.0
X1 = 1.0
Y0 = 0.0
Y1 = 1.0

n_data = 20000
t_data = torch.rand(n_data) * (T1 - T0) + T0
x_data = torch.rand(n_data) * (X1 - X0) + X0
y_data = torch.rand(n_data) * (Y1 - Y0) + Y0
txy_data = torch.stack((t_data, x_data, y_data), dim=1)
u_data = heat(t_data, x_data, y_data).unsqueeze(1)

k_est = torch.randn((1,), requires_grad=True)

criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS((*list(model.parameters()), k_est), lr=1, max_iter=10000, line_search_fn="strong_wolfe")

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
    u_pred = model(txy_data)
    loss = criterion(u_pred, u_data)

    u_t = model_jacobian(txy_data)[:, 0]
    u_hessian = model_hessian(txy_data)[:, 1:, 1:]
    u_nabla = functorch.vmap(torch.trace)(u_hessian)
    f = u_t - (k_est**2) * u_nabla
    loss += 0.0001 * torch.mean(f**2)

    loss.backward()
    print(f"Epoch: {epoch:4d}, Loss: {loss.item():.6f}, k: {k_est.item():.6f}")

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
    y = torch.linspace(Y0, Y1, n)
    tt, xx, yy = torch.meshgrid(t, x, y, indexing="xy")
    ttxxyy = torch.stack((tt.flatten(), xx.flatten(), yy.flatten()), dim=1)
    uu = model(ttxxyy)
    un = torch.reshape(uu, tt.shape)
    uu_true = heat(tt, xx, yy)

    print("True difference:", torch.mean((un - uu_true)**2))

    plt.figure()
    print(T0)
    plt.pcolormesh(x, y, un[:, 0, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.xlabel(r"$y$")
    plt.tight_layout()
    plt.savefig("heat2d_1_discovery.pdf")

    plt.figure()
    print((T1 - T0) / 3)
    plt.pcolormesh(x, y, un[:, n // 3 - 1, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.xlabel(r"$y$")
    plt.tight_layout()
    plt.savefig("heat2d_2_discovery.pdf")

    plt.figure()
    print((T1 - T0) * 2 / 3)
    plt.pcolormesh(x, y, un[:, n * 2 // 3 - 1, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.xlabel(r"$y$")
    plt.tight_layout()
    plt.savefig("heat2d_3_discovery.pdf")

    plt.figure()
    print(T1)
    plt.pcolormesh(x, y, un[:, n - 1, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.xlabel(r"$y$")
    plt.tight_layout()
    plt.savefig("heat2d_4_discovery.pdf")
