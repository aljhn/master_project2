import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
delta_T = T1 / T_iterations
T_now = T0
X0 = -1.0
X1 = 1.0

# input_dim = 2
output_dim = 1
hidden_dim = 20
layers = 5
L = X1 - X0
m = 5
input_dim = 2 * m + 1 + 1

fourier_omegas = torch.arange(1, m + 1, 1, device=device).unsqueeze(0) * 2.0 * np.pi / L

def fourier_embedding(x):
    embedding = torch.cat((torch.ones_like(x, device=device), torch.cos(x @ fourier_omegas), torch.sin(x @ fourier_omegas)), dim=1)
    return embedding

alpha = 5
beta = 0.5
gamma = 0.005

n_i = 1000

t_i = torch.ones((n_i, 1), device=device) * T0
x_i = torch.rand((n_i, 1), device=device) * (X1 - X0) + X0
tx_i = torch.cat((t_i, x_i), dim=1)

t_ii = torch.ones((n_i, 1), device=device) * (T_now + delta_T)
tx_ii = torch.cat((t_i, x_i), dim=1)

n_pinn = 100

n_t = 10

criterion = nn.MSELoss()

max_epochs = 2
epoch = 0


beta_i = 1e3
beta_f = 1.0

delta = 0.99
epsilon_list = [1e-2, 1e-1, 1.0, 1e1, 1e2]

i_losses = []
f_losses = []

def closure():
    global epoch
    epoch += 1

    if epoch > max_epochs:
        raise KeyboardInterrupt

    optimizer.zero_grad()

    loss_weightings = torch.ones(n_t, device=device)
    losses = torch.zeros(n_t, device=device)

    u_pred = model(tx_i)
    initial_loss = criterion(u_pred, u_i)
    losses[0] = beta_i * initial_loss

    loss_sum = losses[0]


    T0_now = T_now
    delta_T_now = delta_T / n_t

    for i in range(1, n_t):
        t_min = T0_now + delta_T_now * (i - 1)
        t_max = T0_now + delta_T_now * i
        t_pinn = torch.rand((n_pinn, 1), device=device) * (t_max - t_min) + t_min
        x_pinn = torch.rand((n_pinn, 1), device=device) * (X1 - X0) + X0
        tx_pinn = torch.cat((t_pinn, x_pinn), dim=1)
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

        f = u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx

        loss_weightings[i] = torch.exp(-epsilon * loss_sum)
        losses[i] = beta_f * torch.mean(f**2)
        loss_sum += losses[i]

    loss = torch.mean(loss_weightings * losses)
    loss.backward()

    print(f"Time march: {int((T_now - T0) / delta_T) + 1}/{T_iterations}, Epoch: {epoch:5d}, Epsilon: {epsilon:.3f}, Loss: {loss.item():.6f}")

    if torch.min(loss_weightings[1:]) > delta:
        raise KeyboardInterrupt

    return loss


output_list = []

t_start = 0
files = os.listdir("./")
for i in range(T_iterations):
    if f"ks_{i}.pth" in files:
        t_start = i + 1

t_iteration_range = range(t_start, T_iterations)

for t_iteration in t_iteration_range: # Time-marching
    if t_iteration == 0:
        u_i = initial_condition(x_i)
    else:
        with torch.no_grad():
            u_i = model(tx_ii)

    model = ModifiedMLP(input_dim, output_dim, hidden_dim, layers)
    model.to(device)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=10000, line_search_fn="strong_wolfe")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epsilon in epsilon_list:
        epoch = 0
        while True:
            try:
                optimizer.step(closure)
            except KeyboardInterrupt:
                break

    with torch.no_grad():
        n = 100
        t = torch.linspace(T0, T0 + delta_T, n, device=device)
        x = torch.linspace(X0, X1, n, device=device)
        tt, xx = torch.meshgrid(t, x, indexing="xy")
        ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
        uu = model(ttxx)
        un = torch.reshape(uu, tt.shape)
        np.savetxt(f"ks_{t_iteration}.txt", un)
        output_list.append([un.cpu()])

    torch.save(model.state_dict(), f"ks_{t_iteration}.pth")


with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    # plt.figure()
    # plt.plot(np.arange(1, len(i_losses) + 1), i_losses)
    # plt.plot(np.arange(1, len(f_losses) + 1), f_losses)
    # plt.xlabel(r"Epoch")
    # plt.ylabel(r"Loss")
    # plt.legend(["Initial", "Physics"])
    # plt.yscale("log")
    # plt.tight_layout()
    # plt.savefig("ks_loss.pdf")
    # plt.show()

    # n = 100
    # t = torch.linspace(T0, T1, n, device=device)
    # x = torch.linspace(X0, X1, n, device=device)
    # tt, xx = torch.meshgrid(t, x, indexing="xy")
    # ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    # uu = model(ttxx)
    # un = torch.reshape(uu, tt.shape)
    # plt.figure()
    # plt.pcolormesh(t.cpu(), x.cpu(), un.cpu(), vmin=-2.0, vmax=2.0, cmap="rainbow")
    # plt.colorbar()
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$x$")
    # plt.tight_layout()
    # plt.savefig("ks.pdf")
    # plt.show()
