import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchdiffeq


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def riccati(t, x):
    return x * x - t


T0 = 0.0
T1 = 10.0
t_initial = torch.tensor([[T0]])
x_initial = torch.rand((1, 1)) * 5.0 - 5.0

t_true = torch.linspace(T0, T1, 1000)
x_true = torchdiffeq.odeint(riccati, x_initial, t_true)
t_true = t_true.unsqueeze(1)
x_true = x_true[:, 0, :]

n_pinn = 1000
t_pinn = torch.rand((n_pinn, 1)) * (T1 - T0) + T0
t_pinn.requires_grad = True


model = nn.Sequential(
    nn.Linear(1, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 1)
)

model2 = nn.Sequential(
    nn.Linear(1, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

epochs = 1000
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()

        x_pred = model(t_initial)
        loss = criterion(x_pred, x_initial)

        x_pinn = model(t_pinn)
        dx_pinn = torch.autograd.grad(x_pinn, t_pinn, grad_outputs=torch.ones_like(x_pinn), retain_graph=True, create_graph=True)[0]
        f = dx_pinn - x_pinn**2 + t_pinn
        loss += 0.1 * torch.mean(f**2)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

        optimizer2.zero_grad()

        x_pinn = model2(t_pinn)
        dx_pinn = torch.autograd.grad(x_pinn, t_pinn, grad_outputs=torch.ones_like(x_pinn), retain_graph=True, create_graph=True)[0]
        f = dx_pinn - x_pinn**2 + t_pinn
        loss2 = torch.mean(f**2)
        loss2.backward()
        optimizer2.step()
    except KeyboardInterrupt:
        break


with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    x = model(t_true)
    plt.figure()
    plt.plot(t_true, x_true)
    plt.plot(t_true, x)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["True system", "Learned system"])
    plt.tight_layout()
    plt.savefig("riccati.pdf")
    plt.show()

    x = model2(t_true)
    plt.figure()
    plt.plot(t_true, -(t_true**0.5))
    plt.plot(t_true, x)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["Equilibrium point", "Learned system"])
    plt.tight_layout()
    plt.savefig("riccati_physics.pdf")
    plt.show()

