import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# model = nn.Sequential(
#     nn.Linear(2, 50),
#     nn.Tanh(),
#     nn.Linear(50, 50),
#     nn.Tanh(),
#     nn.Linear(50, 50),
#     nn.Tanh(),
#     nn.Linear(50, 50),
#     nn.Tanh(),
#     nn.Linear(50, 1)
# )


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

        self.activation = nn.Tanh()
        


    def fourier_embedding(x):
        return None


    def forward(self, X):
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
T1 = 0.1
X0 = -1.0
X1 = 1.0

# input_dim = 2
output_dim = 1
hidden_dim = 50
layers = 5
L = X1 - X0
m = 5
input_dim = 2 * m + 1 + 1
model = ModifiedMLP(input_dim, output_dim, hidden_dim, layers)

fourier_omegas = torch.arange(1, m + 1, 1).unsqueeze(0) * 2.0 * np.pi / L

def fourier_embedding(x):
    embedding = torch.cat((torch.ones_like(x), torch.cos(x @ fourier_omegas), torch.sin(x @ fourier_omegas)), dim=1)
    return embedding


alpha = 5
beta = 0.5
gamma = 0.005

n_ib = 200

t_i = torch.ones((n_ib // 2, 1)) * T0
x_i = torch.rand((n_ib // 2, 1)) * (X1 - X0) + X0
u_i = initial_condition(x_i)
x_i = fourier_embedding(x_i)
tx_i = torch.cat((t_i, x_i), dim=1)

# t_b = torch.rand((n_ib // 2, 1)) * (T1 - T0) + T0
# x_b = torch.cat((torch.ones((n_ib // 4, 1)) * X0, torch.ones((n_ib // 4, 1)) * X1), dim=0)
# x_b = fourier_embedding(x_b)
# tx_b = torch.cat((t_b, x_b), dim=1)

# tx_ib = torch.cat((tx_i, tx_b), dim=0)

n_pinn = 2000
t_pinn = torch.rand((n_pinn, 1)) * (T1 - T0) + T0
x_pinn = torch.rand((n_pinn, 1)) * (X1 - X0) + X0
x_pinn = fourier_embedding(x_pinn)
tx_pinn = torch.cat((t_pinn, x_pinn), dim=1)

tx_pinn.requires_grad = True

criterion = nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=10000, line_search_fn="strong_wolfe")

max_epochs = 1000
epoch = 0


beta_i = 1.0
beta_b = 1.0
beta_f = 1.0

i_losses = []
b_losses = []
f_losses = []

def closure():
    global epoch
    epoch += 1

    if epoch > max_epochs:
        raise KeyboardInterrupt

    optimizer.zero_grad()

    # u_pred = model(tx_ib)
    u_pred = model(tx_i)
    u_pred_i = u_pred[:n_ib // 2]
    initial_loss = criterion(u_pred_i, u_i)

    # u_pred_b = u_pred[n_ib // 2:]
    # u_b1 = u_pred_b[:n_ib // 4]
    # u_b2 = u_pred_b[n_ib // 4:]
    # periodic_boundary_loss = criterion(u_b1, u_b2)

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
    physics_loss = torch.mean(f**2)

    # loss = beta_i * initial_loss + beta_b * periodic_boundary_loss + beta_f * physics_loss
    loss = beta_i * initial_loss  + beta_f * physics_loss
    loss.backward()

    # print(f"Epoch: {epoch:5d}, I: {initial_loss.item():.8f}, B: {periodic_boundary_loss.item():.8f}, F: {physics_loss.item():.8f}")
    print(f"Epoch: {epoch:5d}, I: {initial_loss.item():.8f}, F: {physics_loss.item():.8f}")

    i_losses.append(initial_loss.item())
    # b_losses.append(periodic_boundary_loss.item())
    f_losses.append(physics_loss.item())

    return loss
    

while True:
    try:
        optimizer.step(closure)
    except KeyboardInterrupt:
        break


with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    plt.figure()
    plt.plot(np.arange(1, len(i_losses) + 1), i_losses)
    # plt.plot(np.arange(1, len(b_losses) + 1), b_losses)
    plt.plot(np.arange(1, len(f_losses) + 1), f_losses)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Loss")
    # plt.legend(["Initial", "Boundary", "Physics"])
    plt.legend(["Initial", "Physics"])
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("ks_loss.pdf")
    plt.show()

    n = 100
    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    u_shape = tt.shape
    tt = tt.reshape((-1, 1))
    xx = xx.reshape((-1, 1))
    xx = fourier_embedding(xx)
    ttxx = torch.cat((tt, xx), dim=1)
    uu = model(ttxx)
    un = torch.reshape(uu, u_shape)
    plt.figure()
    plt.pcolormesh(t, x, un, vmin=-2.0, vmax=2.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("ks.pdf")
    plt.show()
