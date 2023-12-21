import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
    nn.Linear(50, 1),
)


def solution(x, y):
    return torch.cos(np.pi * x) * torch.sinh(np.pi * y)


X0 = 0.0
X1 = 1.0
Y0 = 0.0
Y1 = 1.0

n_data = 200

x1 = torch.ones(n_data // 4) * X0
y1 = torch.rand(n_data // 4) * (Y1 - Y0) + Y0
u1 = solution(x1, y1)
xy1 = torch.stack((x1, y1), dim=1)

x2 = torch.ones(n_data // 4) * X1
y2 = torch.rand(n_data // 4) * (Y1 - Y0) + Y0
u2 = solution(x2, y2)
xy2 = torch.stack((x2, y2), dim=1)

x3 = torch.rand(n_data // 4) * (X1 - X0) + X0
y3 = torch.ones(n_data // 4) * Y0
u3 = solution(x3, y3)
xy3 = torch.stack((x3, y3), dim=1)

x4 = torch.rand(n_data // 4) * (X1 - X0) + X0
y4 = torch.ones(n_data // 4) * Y1
u4 = solution(x4, y4)
xy4 = torch.stack((x4, y4), dim=1)

xy_data = torch.cat((xy1, xy2, xy3, xy4), dim=0)
u_data = torch.cat((u1, u2, u3, u4), dim=0).unsqueeze(1)

n_pinn = 10000
x_pinn = torch.rand(n_pinn) * (X1 - X0) + X0
y_pinn = torch.rand(n_pinn) * (Y1 - Y0) + Y0
xy_pinn = torch.stack((x_pinn, y_pinn), dim=1)

n = 100
x = torch.linspace(X0, X1, n)
y = torch.linspace(Y0, Y1, n)
xx, yy = torch.meshgrid(x, y, indexing="xy")
xxyy = torch.stack((xx.flatten(), yy.flatten()), dim=1)
uu_true = solution(xx, yy)

criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(
    model.parameters(), lr=1, max_iter=10000, line_search_fn="strong_wolfe"
)

model_hessian = torch.vmap(torch.func.hessian(lambda x: model(x).squeeze()))

max_epochs = 1000
epoch = 0

train_losses = np.zeros(max_epochs)
val_losses = np.zeros(max_epochs)


def closure():
    global epoch
    epoch += 1

    if epoch > max_epochs:
        raise KeyboardInterrupt

    optimizer.zero_grad()

    u_pred = model(xy_data)
    loss = criterion(u_pred, u_data)

    u_hessian = model_hessian(xy_pinn)
    u_laplacian = torch.vmap(torch.trace)(u_hessian)
    f = u_laplacian
    loss += 0.1 * torch.mean(f**2)

    loss.backward()

    with torch.no_grad():
        uu = model(xxyy)
        un = torch.reshape(uu, xx.shape)
        val_loss = torch.mean((un - uu_true) ** 2)

    train_losses[epoch - 1] = loss.item()
    val_losses[epoch - 1] = val_loss.item()

    print(f"Epoch: {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    return loss


while True:
    try:
        optimizer.step(closure)
    except KeyboardInterrupt:
        break

np.savetxt("train_losses", train_losses)
np.savetxt("val_losses", val_losses)

exit()


with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    n = 100
    x = torch.linspace(X0, X1, n)
    y = torch.linspace(Y0, Y1, n)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xxyy = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    uu = model(xxyy)
    un = torch.reshape(uu, xx.shape)
    uu_true = solution(xx, yy)
    print("True difference:", torch.mean((un - uu_true) ** 2))

    plt.figure()
    plt.pcolormesh(x, y, uu_true, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.tight_layout()
    plt.savefig("laplace_forward_true.pdf")
    plt.show()

    plt.figure()
    plt.pcolormesh(x, y, un, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.tight_layout()
    plt.savefig("laplace_forward.pdf")
    plt.show()
