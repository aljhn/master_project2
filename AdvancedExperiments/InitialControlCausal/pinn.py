import sys
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModifiedMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers):
        super(ModifiedMLP, self).__init__()
        self.U_layer = nn.Linear(input_dim, hidden_dim)
        self.V_layer = nn.Linear(input_dim, hidden_dim)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, output_dim)

        self.Z_layers = []
        for i in range(layers - 1):
            self.Z_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.Z_layers = nn.Sequential(*self.Z_layers)

        self.activation = nn.Tanh()

    def forward(self, X):
        T, X = X[:, 0].unsqueeze(1), X[:, 1:]
        FX = fourier_embedding(X)
        X = torch.cat((T, FX), dim=1)

        U = self.activation(self.U_layer(X))
        V = self.activation(self.V_layer(X))

        H = self.activation(self.input_layer(X))
        for Z_layer in self.Z_layers:
            Z = self.activation(Z_layer(H))
            H = (1 - Z) * U + Z * V

        return self.final_layer(H)


def initial_condition(x):
    return (x**2.0) * torch.cos(np.pi * x)


T0 = 0.0
T1 = 5.0
X0 = 0.0
X1 = 4.0

nu = 0.01


def solution(t, x):
    return (
        2.0
        * nu
        * np.pi
        * torch.exp(-(np.pi**2.0) * nu * (t - 5.0))
        * torch.sin(np.pi * x)
        / (2.0 + torch.exp(-(np.pi**2.0) * nu * (t - 5.0)) * torch.cos(np.pi * x))
    )


output_dim = 1
hidden_dim = 50
layers = 5
L = X1 - X0
m = 5
input_dim = 2 * m + 1 + 1

fourier_omegas = torch.arange(1, m + 1, 1, device=device).unsqueeze(0) * 2.0 * np.pi / L


def fourier_embedding(x):
    embedding = torch.cat(
        (
            torch.ones_like(x, device=device),
            torch.cos(x @ fourier_omegas),
            torch.sin(x @ fourier_omegas),
        ),
        dim=1,
    )
    return embedding


n_i = 1000

t_i = torch.ones((n_i, 1), device=device) * T0
x_i = torch.rand((n_i, 1), device=device) * (X1 - X0) + X0
tx_i = torch.cat((t_i, x_i), dim=1)

n_t = 20
n_x = 500

weighting_matrix = torch.triu(torch.ones((n_t, n_t), device=device), diagonal=1).T

criterion = nn.MSELoss()

max_epochs = 20000

beta_i = 1.0
beta_f = 1.0

delta = 0.99
epsilon_list = [1e-2, 1e-1, 1.0, 1e1, 1e2]

n_j = 41
h_j = 1.0 / n_j
x_j = torch.linspace(X0, X1, n_j, device=device)
u_j = solution(torch.tensor([T1]), x_j)
t_j = torch.ones(n_j, device=device) * T1
tx_j = torch.stack((t_j, x_j), dim=1)


def trapezoid(f, h):
    return h * (torch.sum(f[1:-1]) + (f[0] + f[-1]) / 2.0)


model_u = ModifiedMLP(input_dim, output_dim, hidden_dim, layers)
model_u.to(device)

model_c = nn.Sequential(
    nn.Linear(1, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 1),
)
model_c.to(device)

optimizer = torch.optim.Adam((*model_u.parameters(), *model_c.parameters()), lr=1e-3)


def closure():
    optimizer.zero_grad()

    u_i_pred = model_u(tx_i)
    u_i = model_c(x_i)
    initial_loss = criterion(u_i_pred, u_i)
    losses_0 = beta_i * initial_loss

    t_pinn = torch.rand((n_t,), device=device) * (T1 - T0) + T0
    t_pinn, _ = torch.sort(t_pinn)
    x_pinn = torch.rand((n_x,), device=device) * (X1 - X0) + X0
    tt_pinn, xx_pinn = torch.meshgrid((t_pinn, x_pinn), indexing="xy")
    tx_pinn = torch.stack((tt_pinn.flatten(), xx_pinn.flatten()), dim=1)
    tx_pinn.requires_grad = True

    u = model_u(tx_pinn)[:, 0]
    u_grad = torch.autograd.grad(
        u,
        tx_pinn,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_t = u_grad[:, 0]
    u_x = u_grad[:, 1]
    u_x_grad = torch.autograd.grad(
        u_x,
        tx_pinn,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_xx = u_x_grad[:, 1]

    u = torch.reshape(u, (n_t, n_x))
    u_t = torch.reshape(u_t, (n_t, n_x))
    u_x = torch.reshape(u_x, (n_t, n_x))
    u_xx = torch.reshape(u_xx, (n_t, n_x))

    f = u_t + u * u_x - nu * u_xx
    losses = beta_f * torch.mean(f**2, dim=1)

    with torch.no_grad():
        loss_weightings = torch.exp(-epsilon * (weighting_matrix @ losses + losses_0))

    loss = torch.mean(loss_weightings * losses + losses_0)

    u_j_pred = model_u(tx_j)[:, 0]
    j = 0.5 * ((u_j_pred - u_j) ** 2.0)
    cost = trapezoid(j, h_j)
    loss += cost

    loss.backward()

    if torch.min(loss_weightings) > delta:
        raise ValueError("stop")

    return loss


for epsilon in epsilon_list:
    epoch = 0
    pbar = tqdm(range(max_epochs))
    for i_epoch in pbar:
        try:
            loss = optimizer.step(closure)
            pbar.set_postfix(
                {"Epsilon": f"{epsilon:.3f}", "Loss": f"{loss.item():.6f}"}
            )

        except KeyboardInterrupt:
            sys.exit()
        except ValueError as e:
            if str(e) == "stop":
                break


torch.save(model_u.state_dict(), "model_u.pth")
torch.save(model_c.state_dict(), "model_c.pth")
