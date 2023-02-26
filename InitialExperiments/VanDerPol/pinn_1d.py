import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import functorch
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


t_initial = torch.tensor([[0.0]])
x_initial = torch.rand((1, 2)) * 5.0 - 5.0

t_true = torch.linspace(0, 20, 1000)
x_true = torchdiffeq.odeint(vanderpol, x_initial, t_true)
t_true = t_true.unsqueeze(1)
x_true = x_true[:, 0, :]

t_initial = t_true[::100, :]
x_initial = x_true[::100, 0].unsqueeze(1)

n_pinn = 100
t_pinn = torch.linspace(0.0, 20.0, n_pinn).unsqueeze(1)
# t_pinn = torch.rand((n_pinn, 1)) * 20.0
t_pinn.requires_grad = True

model = nn.Sequential(
    nn.Linear(1, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# model_jacobian = functorch.vmap(functorch.jacfwd(lambda x: model(x).squeeze()))
# model_hessian = functorch.vmap(functorch.hessian(lambda x: model(x).squeeze()))

epochs = 10000
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()
        x_pred = model(t_initial)
        loss = criterion(x_pred, x_initial)

        # x_pinn = model(t_pinn)[:, 0]
        # dx_pinn = model_jacobian(t_pinn)[:, 0]
        # ddx_pinn = model_hessian(t_pinn)[:, 0, 0]

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
    x = model(t_true)
    plt.figure()
    plt.plot(t_true, x_true[:, 0])
    plt.plot(t_true, x[:, 0])
    plt.scatter(t_initial, x_initial)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["True system", "Learned system"])
    plt.title("Van der Pol Oscillator")
    plt.savefig("vdp.pdf")
    plt.show()
