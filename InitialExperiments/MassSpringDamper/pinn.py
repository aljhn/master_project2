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


m = 1.0
k = 500.0
d = 5.0
def massspringdamper(t, x):
    x1, x2 = x
    dx1 = x2
    dx2 = - k / m * x1 - d / m * x2
    return torch.stack((dx1, dx2), dim=0)


T1 = 1.0
x_initial = torch.tensor([1.0, 0.0])

n_data = 500
t_true = torch.linspace(0.0, T1, n_data)
x_true = torchdiffeq.odeint(massspringdamper, x_initial, t_true)
x_true = x_true[:, 0]

t_data = t_true[:200:20].unsqueeze(1)
x_data = x_true[:200:20].unsqueeze(1)
# t_data = t_true[0, None].unsqueeze(1)
# x_data = x_true[0, None].unsqueeze(1)

n_pinn = 30
t_pinn = torch.linspace(0.0, T1, n_pinn).unsqueeze(1)
t_pinn.requires_grad = True

model = nn.Sequential(
    nn.Linear(1, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 40),
    nn.Tanh(),
    nn.Linear(40, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# model_jacobian = functorch.vmap(functorch.jacfwd(lambda t: model(t).squeeze()))
# model_hessian = functorch.vmap(functorch.hessian(lambda t: model(t).squeeze()))

epochs = 20000
train_losses = np.zeros(epochs)
for epoch in range(1, epochs + 1):
    try:
        optimizer.zero_grad()
        x_pred = model(t_data)
        loss = criterion(x_pred, x_data)

        x_pinn = model(t_pinn)
        # dx_pinn = model_jacobian(t_pinn)
        # ddx_pinn = model_hessian(t_pinn)[:, :, 0]

        dx_pinn = torch.autograd.grad(x_pinn, t_pinn, grad_outputs=torch.ones_like(x_pinn), retain_graph=True, create_graph=True)[0]
        ddx_pinn = torch.autograd.grad(dx_pinn, t_pinn, grad_outputs=torch.ones_like(dx_pinn), retain_graph=True, create_graph=True)[0]

        f = m * ddx_pinn + d * dx_pinn + k * x_pinn
        loss += 0.0001 * torch.mean(f**2)

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        train_losses[epoch - 1] = loss.item()
    except KeyboardInterrupt:
        break


with torch.no_grad():
    x = model(t_true.unsqueeze(1))
    plt.figure()
    plt.plot(t_true, x_true)
    plt.plot(t_true, x)
    plt.scatter(t_data, x_data)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["True system", "Learned system"])
    plt.title("Mass Spring Damper")
    plt.savefig("msd.pdf")
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Train Loss")
    plt.savefig("msd_loss.pdf")
    plt.show()

