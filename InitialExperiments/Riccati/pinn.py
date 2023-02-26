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


def riccati(t, x):
    return x * x - t


t_initial = torch.tensor([[0.0]])
x_initial = torch.rand((1, 1)) * 5.0 - 5.0

t_true = torch.linspace(0, 10, 1000)
x_true = torchdiffeq.odeint(riccati, x_initial, t_true)
x_true = x_true[:, 0, :]

n_pinn = 1000
t_pinn = torch.linspace(0.0, 10.0, n_pinn).unsqueeze(1)
t_pinn.requires_grad = True


model = nn.Sequential(
    nn.Linear(1, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 50),
    nn.Tanh(),
    nn.Linear(50, 1)
)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# model_jacobian = functorch.vmap(functorch.grad(lambda x: model(x).squeeze()))

epochs = 1000
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()

        x_pred = model(t_initial)
        loss = criterion(x_pred, x_initial)

        x_pinn = model(t_pinn)
        # dx_pinn = model_jacobian(t_pinn)
        dx_pinn = torch.autograd.grad(x_pinn, t_pinn, grad_outputs=torch.ones_like(x_pinn), retain_graph=True, create_graph=True)[0]
        f = dx_pinn - (x_pinn**2) + t_pinn
        loss += 0.1 * torch.mean(f**2)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    except KeyboardInterrupt:
        break


with torch.no_grad():
    x = model(t_true.unsqueeze(1))
    plt.figure()
    plt.plot(t_true, x_true)
    plt.plot(t_true, x)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["True system", "Learned system"])
    plt.title("Riccati Equation")
    plt.savefig("riccati.pdf")
    plt.show()
