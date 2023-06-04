import os
import sys
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
    return -torch.sin(np.pi * x)


T0 = 0.0
T1 = 1.0
T_iterations = 10
delta_T = (T1 - T0) / T_iterations
X0 = -1.0
X1 = 1.0

output_dim = 1
hidden_dim = 50
layers = 5
L = X1 - X0
m = 5
input_dim = 2 * m + 1 + 1

fourier_omegas = torch.arange(1, m + 1, 1, device=device).unsqueeze(0) * 2.0 * np.pi / L

def fourier_embedding(x):
    embedding = torch.cat((torch.ones_like(x, device=device), torch.cos(x @ fourier_omegas), torch.sin(x @ fourier_omegas)), dim=1)
    return embedding

alpha = 5.0
beta = 0.5
gamma = 0.005

n_i = 1000

t_i = torch.ones((n_i, 1), device=device) * T0
x_i = torch.rand((n_i, 1), device=device) * (X1 - X0) + X0
tx_i = torch.cat((t_i, x_i), dim=1)

t_ii = torch.ones((n_i, 1), device=device) * (T0 + delta_T)
tx_ii = torch.cat((t_ii, x_i), dim=1)

n_t = 20
n_x = 100

weighting_matrix = torch.triu(torch.ones((n_t, n_t), device=device), diagonal=1).T

criterion = nn.MSELoss()

max_epochs = 10000

beta_i = 1e3
beta_f = 1.0

delta = 0.99
epsilon_list = [1e-2, 1e-1, 1.0, 1e1, 1e2]

i_losses = []
f_losses = []

def closure():
    optimizer.zero_grad()

    u_pred = model(tx_i)
    initial_loss = criterion(u_pred, u_i)
    losses_0 = beta_i * initial_loss

    t_pinn = torch.rand((n_t,), device=device) * (T1 - T0) + T0
    t_pinn, _ = torch.sort(t_pinn)
    x_pinn = torch.rand((n_x,), device=device) * (X1 - X0) + X0
    tt_pinn, xx_pinn = torch.meshgrid((t_pinn, x_pinn), indexing="xy")
    tx_pinn = torch.stack((tt_pinn.flatten(), xx_pinn.flatten()), dim=1)
    tx_pinn.requires_grad = True

    u = model(tx_pinn)[:, 0]
    u_grad = torch.autograd.grad(u, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_t = u_grad[:, 0]
    u_x = u_grad[:, 1]
    u_x_grad = torch.autograd.grad(u_x, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = u_x_grad[:, 1]
    u_xx_grad = torch.autograd.grad(u_xx, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xxx = u_xx_grad[:, 1]
    u_xxx_grad = torch.autograd.grad(u_xxx, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xxxx = u_xxx_grad[:, 1]

    u = torch.reshape(u, (n_t, n_x))
    u_t = torch.reshape(u_t, (n_t, n_x))
    u_x = torch.reshape(u_x, (n_t, n_x))
    u_xx = torch.reshape(u_xx, (n_t, n_x))
    u_xxxx = torch.reshape(u_xxxx, (n_t, n_x))

    f = u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx
    losses = beta_f * torch.mean(f**2, dim=1)

    with torch.no_grad():
        loss_weightings = torch.exp(-epsilon * (weighting_matrix @ losses + losses_0))

    loss = torch.mean(loss_weightings * losses + losses_0)
    loss.backward()

    if torch.min(loss_weightings) > delta:
        raise ValueError("stop")
    return loss


output_list = []

t_start = 0
epoch_start = 0
epsilon_start = 0
checkpoint = False

files = os.listdir("./")
if "checkpoint.txt" in files:
    with open("checkpoint.txt", "r") as f:
        lines = f.readlines()
        t_start = int(lines[0])
        epoch_start = int(lines[1])
        epsilon_start = int(lines[2])
        checkpoint = True

t_iteration_range = range(t_start, T_iterations)

for t_iteration in t_iteration_range:
    if t_iteration == 0:
        u_i = initial_condition(x_i)
    else:
        with torch.no_grad():
            model_prev = ModifiedMLP(input_dim, output_dim, hidden_dim, layers)
            model_prev.load_state_dict(torch.load(f"model_{t_iteration - 1}.pth"))
            model_prev.to(device)
            u_i = model_prev(tx_ii)

    model = ModifiedMLP(input_dim, output_dim, hidden_dim, layers)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if checkpoint:
        model.load_state_dict(torch.load("model_checkpoint.pth"))
        optimizer.load_state_dict(torch.load("optimizer_checkpoint.pth"))

    for i_epsilon in range(epsilon_start, len(epsilon_list)):
        epsilon = epsilon_list[i_epsilon]
        epoch = 0
        pbar = tqdm(range(epoch_start, max_epochs))
        for i_epoch in pbar:
            try:
                loss = optimizer.step(closure)
                pbar.set_postfix({"Time march": f"{t_iteration + 1}/{T_iterations}", "Epsilon": f"{epsilon:.3f}", "Loss": f"{loss.item():.6f}"})

                if i_epoch % 100 == 0:
                    torch.save(model.state_dict(), "model_checkpoint.pth")
                    torch.save(optimizer.state_dict(), "optimizer_checkpoint.pth")
                    with open("checkpoint.txt", "w") as f:
                        f.write(f"{t_iteration}\n{i_epoch}\n{i_epsilon}")

            except KeyboardInterrupt:
                torch.save(model.state_dict(), "model_checkpoint.pth")
                torch.save(optimizer.state_dict(), "optimizer_checkpoint.pth")
                with open("checkpoint.txt", "w") as f:
                    f.write(f"{t_iteration}\n{i_epoch}\n{i_epsilon}")
                sys.exit()
            except ValueError as e:
                if str(e) == "stop":
                    break

        epoch_start = 0
    epsilon_start = 0

    with torch.no_grad():
        n = 100
        t = torch.linspace(T0, T0 + delta_T, n, device=device)
        x = torch.linspace(X0, X1, n, device=device)
        tt, xx = torch.meshgrid(t, x, indexing="xy")
        ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
        uu = model(ttxx)
        un = torch.reshape(uu, tt.shape)
        np.savetxt(f"u_{t_iteration}.txt", un.cpu())
        output_list.append([un.cpu()])

    torch.save(model.state_dict(), f"model_{t_iteration}.pth")
