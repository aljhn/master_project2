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


mu = 1.0
def vanderpol(t, x):
    x1 = x[0, 0]
    x2 = x[0, 1]
    dx1 = x2
    dx2 = mu * (1.0 - (x1**2.0)) * x2 - x1
    return torch.stack((dx1, dx2), dim=0)


T0 = 0.0
T1 = 20.0
t_initial = torch.tensor([[T0]])
x_initial = torch.rand((1, 2)) * 5.0 - 5.0

t_true = torch.linspace(T0, T1, 1000)
x_true = torchdiffeq.odeint(vanderpol, x_initial, t_true)
t_true = t_true.unsqueeze(1)
x_true = x_true[:, 0, 0].unsqueeze(1)

t_data = t_true[::100, :]
x_data = x_true[::100, :]

n_pinn = 100
t_pinn = torch.rand((n_pinn, 1)) * (T1 - T0) + T0
t_pinn.requires_grad = True

model_reg = nn.Sequential(
    nn.Linear(1, 20),
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

model = nn.Sequential(
    nn.Linear(1, 20),
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

criterion = nn.MSELoss()
optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10000
for epoch in range(1, epochs + 1):
    try:
        optimizer_reg.zero_grad()
        x_pred = model_reg(t_data)
        loss_reg = criterion(x_pred, x_data)
        loss_reg.backward()
        optimizer_reg.step()

        optimizer.zero_grad()
        x_pred = model(t_data)
        loss = criterion(x_pred, x_data)

        x_pinn = model(t_pinn)
        dx_pinn = torch.autograd.grad(x_pinn, t_pinn, grad_outputs=torch.ones_like(x_pinn), retain_graph=True, create_graph=True)[0]
        ddx_pinn = torch.autograd.grad(dx_pinn, t_pinn, grad_outputs=torch.ones_like(dx_pinn), retain_graph=True, create_graph=True)[0]
        f = ddx_pinn - (mu * (1.0 - (x_pinn**2.0)) * dx_pinn - x_pinn)
        loss += 0.1 * torch.mean(f**2)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    except KeyboardInterrupt:
        break


with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    x_reg = model_reg(t_true)
    x = model(t_true)
    plt.figure()
    plt.plot(t_true, x_true)
    plt.plot(t_true, x_reg)
    plt.plot(t_true, x)
    plt.scatter(t_data, x_data)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["True Output", "Neural Network Output", "PINN Output"])
    plt.tight_layout()
    plt.savefig("vdp.pdf")
    plt.show()
