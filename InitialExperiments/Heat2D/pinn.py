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


def heat(t, x, y):
    return torch.sin(3.14 * x) * torch.sin(3.14 * y) * torch.exp(-(3.14**2.0) * t)


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

batch_size = 100

T0 = 0.0
T1 = 0.2
X0 = 0.0
X1 = 1.0
Y0 = 0.0
Y1 = 1.0

t_i = torch.ones((batch_size // 2, 1)) * T0
x_i = torch.rand((batch_size // 2, 1)) * (X1 - X0) + X0
y_i = torch.rand((batch_size // 2, 1)) * (Y1 - Y0) + Y0
u_i = heat(t_i, x_i, y_i)
txy_i = torch.cat((t_i, x_i, y_i), dim=1)

t_b = torch.rand((batch_size // 2, 1)) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((batch_size // 4, 1)) * X0, torch.ones((batch_size // 4, 1)) * X1), dim=0)
y_b = torch.cat((torch.ones((batch_size // 4, 1)) * Y0, torch.ones((batch_size // 4, 1)) * Y1), dim=0)
txy_b = torch.cat((t_b, x_b, y_b), dim=1)
u_b = heat(t_b, x_b, y_b)

txy_ib = torch.cat((txy_i, txy_b), dim=0)
u_ib = torch.cat((u_i, u_b), dim=0)

n_pinn = 10000
t_pinn = torch.rand(n_pinn) * (T1 - T0) + T0
x_pinn = torch.rand(n_pinn) * (X1 - X0) + X0
y_pinn = torch.rand(n_pinn) * (Y1 - Y0) + Y0
txy_pinn = torch.stack((t_pinn, x_pinn, y_pinn), dim=1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model_jacobian = functorch.vmap(functorch.grad(lambda x: model(x).squeeze()))
model_hessian = functorch.vmap(functorch.hessian(lambda x: model(x).squeeze()))

epochs = 1000
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()
        u_pred = model(txy_ib)
        loss = criterion(u_pred, u_ib)

        u_t = model_jacobian(txy_pinn)[:, 0]
        u_hessian = model_hessian(txy_pinn)[:, 1:, 1:]
        u_nabla = functorch.vmap(torch.trace)(u_hessian)
        f = u_t - u_nabla
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
    y = torch.linspace(Y0, Y1, n)
    tt, xx, yy = torch.meshgrid(t, x, y, indexing="xy")
    ttxxyy = torch.stack((tt.flatten(), xx.flatten(), yy.flatten()), dim=1)
    uu = model(ttxxyy)
    un = torch.reshape(uu, tt.shape)
    uu_true = heat(tt, xx, yy)

    print("True difference:", torch.mean((un - uu_true)**2))

    plt.figure()
    print(T0)
    plt.pcolormesh(x, y, un[0, :, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_1.pdf")

    plt.figure()
    print((T1 - T0) / 3)
    plt.pcolormesh(x, y, un[n // 3 - 1, :, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_2.pdf")

    plt.figure()
    print((T1 - T0) * 2 / 3)
    plt.pcolormesh(x, y, un[n * 2 // 3 - 1, :, ], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_3.pdf")

    plt.figure()
    print(T1)
    plt.pcolormesh(x, y, un[n - 1, :, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_4.pdf")


    plt.figure()
    plt.pcolormesh(x, y, uu_true[0, :, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_1_true.pdf")

    plt.figure()
    plt.pcolormesh(x, y, uu_true[n // 3 - 1, :, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_2_true.pdf")

    plt.figure()
    plt.pcolormesh(x, y, uu_true[n * 2 // 3 - 1, :, ], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_3_true.pdf")

    plt.figure()
    plt.pcolormesh(x, y, uu_true[n - 1, :, :], vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("heat2d_4_true.pdf")
