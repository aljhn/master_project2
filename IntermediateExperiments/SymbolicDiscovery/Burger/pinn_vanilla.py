import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
model.to(device)


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

t_i = torch.ones((n_ib // 2, 1), device=device) * T0
x_i = torch.rand((n_ib // 2, 1), device=device) * (X1 - X0) + X0
tx_i = torch.cat((t_i, x_i), dim=1)
u_i = initial_condition(x_i)

t_b = torch.rand((n_ib // 2, 1), device=device) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((n_ib // 4, 1), device=device) * X0, torch.ones((n_ib // 4, 1), device=device) * X1), dim=0)
tx_b = torch.cat((t_b, x_b), dim=1)
u_b = boundary_condition(t_b)

tx_ib = torch.cat((tx_i, tx_b), dim=0)
u_ib = torch.cat((u_i, u_b), dim=0)

n_pinn = 10000
t_pinn = torch.rand(n_pinn, device=device) * (T1 - T0) + T0
x_pinn = torch.rand(n_pinn, device=device) * (X1 - X0) + X0
tx_pinn = torch.stack((t_pinn, x_pinn), dim=1)

# tx_pinn.requires_grad = True

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model_jacobian = torch.vmap(torch.func.grad(lambda x: model(x).squeeze()))
model_hessian = torch.vmap(torch.func.hessian(lambda x: model(x).squeeze()))

max_epochs = 10000

beta_i = 1e2
beta_f = 1.0


def closure():
    optimizer.zero_grad()

    u_pred = model(tx_ib)
    initial_loss = criterion(u_pred, u_ib)

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
    residual_loss = torch.mean(f**2)
    loss = beta_i * initial_loss + beta_f * residual_loss

    loss.backward()
    return loss


epoch_start = 0
losses = []

files = os.listdir("./")
if "checkpoint.txt" in files:
    with open("checkpoint.txt", "r") as f:
        epoch_start = int(f.read())

    model.load_state_dict(torch.load("model_checkpoint.pth"))
    optimizer.load_state_dict(torch.load("optimizer_checkpoint.pth"))
    losses = list(np.loadtxt("losses.txt"))

pbar = tqdm(range(epoch_start, max_epochs))
for epoch in pbar:
    try:
        loss = optimizer.step(closure)
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        losses.append(loss.item())

        if epoch % 100 == 0:
            torch.save(model.state_dict(), "model_checkpoint.pth")
            torch.save(optimizer.state_dict(), "optimizer_checkpoint.pth")
            with open("checkpoint.txt", "w") as f:
                f.write(str(epoch))
            np.savetxt("losses.txt", np.array(losses))


    except KeyboardInterrupt:
        torch.save(model.state_dict(), "model_checkpoint.pth")
        torch.save(optimizer.state_dict(), "optimizer_checkpoint.pth")
        with open("checkpoint.txt", "w") as f:
            f.write(str(epoch))
        np.savetxt("losses.txt", np.array(losses))
        sys.exit()

torch.save(model.state_dict(), f"model.pth")
np.savetxt("losses.txt", np.array(losses))
sys.exit()

with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

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
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("burger.pdf")
    plt.show()

    plt.figure()
    print(T0)
    plt.plot(x, un[:, 0])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice1.pdf")

    plt.figure()
    print((T1 - T0) / 3)
    plt.plot(x, un[:, n // 3 - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice2.pdf")

    plt.figure()
    print((T1 - T0) * 2 / 3)
    plt.plot(x, un[:, n * 2 // 3 - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice3.pdf")

    plt.figure()
    print(T1)
    plt.plot(x, un[:, n - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice4.pdf")
