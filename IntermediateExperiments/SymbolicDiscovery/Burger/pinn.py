import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn as nn
from tqdm import tqdm


seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModifiedMLP(nn.Module):


    def __init__(self, m, L, output_dim, hidden_dim, layers):
        super(ModifiedMLP, self).__init__()
        input_dim = 2 * m + 2
        self.U_layer = nn.Linear(input_dim, hidden_dim)
        self.V_layer = nn.Linear(input_dim, hidden_dim)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, output_dim)

        self.Z_layers = []
        for i in range(layers - 1):
            self.Z_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.Z_layers = nn.Sequential(*self.Z_layers)

        self.activation = nn.Tanh()
        
        self.fourier_omegas = torch.arange(1, m + 1, 1, device=device).unsqueeze(0) * 2.0 * np.pi / L


    def fourier_embedding(self, x):
        embedding = torch.cat((torch.ones_like(x), torch.cos(x @ self.fourier_omegas), torch.sin(x @ self.fourier_omegas)), dim=1)
        return embedding


    def forward(self, X):
        T, X = X[:, 0].unsqueeze(1), X[:, 1:]
        FX = self.fourier_embedding(X)
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


def boundary_condition(t):
    return torch.zeros_like(t)


T0 = 0.0
T1 = 1.0
X0 = -1.0
X1 = 1.0

output_dim = 1
hidden_dim = 50
layers = 5
L = X1 - X0
m = 5

model_true = ModifiedMLP(m, L, output_dim, hidden_dim, layers)
model_true.load_state_dict(torch.load("model_vanilla.pth"))
model_true.to(device)

model_u = nn.Sequential(
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
model_u.to(device)

"""model_b = nn.Sequential(
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
model_b.to(device)"""

model_f = nn.Sequential(
    nn.Linear(3, 20),
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
model_f.to(device)

nu = 0.01 / np.pi

"""
# Compare true vanilla output with some more robust data
data = scipy.io.loadmat("../../../Data/burger.mat")
t_data = torch.tensor(data["t"], device=device, dtype=torch.float32).squeeze()
x_data = torch.tensor(data["x"], device=device, dtype=torch.float32).squeeze()
u_data = torch.tensor(data["usol"], device=device, dtype=torch.float32)
tt_data, xx_data = torch.meshgrid(t_data, x_data, indexing="xy")
ttxx_data = torch.stack((tt_data.flatten(), xx_data.flatten()), dim=1)
uu_pred = model_true(ttxx_data)
u_pred = torch.reshape(uu_pred, u_data.shape)
print(torch.mean((u_pred - u_data)**2))
# Before with vanilla approach: 2e-4
# After with all improvements except time marching: 2.8914e-5
exit()
"""


n_i = 1000
t_i = torch.rand(n_i, device=device) * (T1 - T0) + T0
x_i = torch.rand(n_i, device=device) * (X1 - X0) + X0
tx_i = torch.stack((t_i, x_i), dim=1)
with torch.no_grad():
    u_i = model_true(tx_i)

"""n_b = 1000
t_b = torch.rand((n_b, 1), device=device) * (T1 - T0) + T0
x_b = torch.cat((torch.ones((n_b // 2, 1), device=device) * X0, torch.ones((n_b // 2, 1), device=device) * X1), dim=0)
tx_b = torch.cat((t_b, x_b), dim=1)
u_b = boundary_condition(tx_b)"""

n_pinn = 10000

criterion = nn.MSELoss()
# optimizer = torch.optim.Adam([*model_u.parameters(), *model_b.parameters(), *model_f.parameters()], lr=1e-3)
optimizer = torch.optim.Adam((*model_u.parameters(), *model_f.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

max_epochs = 10000

beta_i = 1e3
# beta_b = 1.0
beta_f = 1.0


def closure():
    optimizer.zero_grad()

    u_i_pred = model_u(tx_i)
    loss_i = criterion(u_i_pred, u_i)

    t_pinn = torch.rand(n_pinn, device=device) * (T1 - T0) + T0
    x_pinn = torch.rand(n_pinn, device=device) * (X1 - X0) + X0
    tx_pinn = torch.stack((t_pinn, x_pinn), dim=1)
    tx_pinn.requires_grad = True

    u = model_u(tx_pinn)[:, 0]
    u_grad = torch.autograd.grad(u, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_t = u_grad[:, 0]
    u_x = u_grad[:, 1]
    u_x_grad = torch.autograd.grad(u_x, tx_pinn, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = u_x_grad[:, 1]

    f = u_t - nu * u_xx # + u * u_x
    u_collection = torch.stack((u, u_t, u_x), dim=1)#.detach()
    f -= model_f(u_collection).squeeze()
    loss_f = torch.mean(f**2)

    # loss = beta_i * loss_i + beta_b * loss_b + beta_f * loss_f
    loss = beta_i * loss_i + beta_f * loss_f
    loss.backward()
    return loss


epoch_start = 0

files = os.listdir("./")
if "checkpoint.txt" in files:
    with open("checkpoint.txt", "r") as f:
        epoch_start = int(f.read())

    model_u.load_state_dict(torch.load("model_u_checkpoint.pth"))
    # model_b.load_state_dict(torch.load("model_b_checkpoint.pth"))
    model_f.load_state_dict(torch.load("model_f_checkpoint.pth"))
    optimizer.load_state_dict(torch.load("optimizer_checkpoint.pth"))

pbar = tqdm(range(epoch_start, max_epochs))
for epoch in pbar:
    try:
        loss = optimizer.step(closure)
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        if epoch % 5000 == 0:
            scheduler.step()

        if epoch % 100 == 0:
            torch.save(model_u.state_dict(), "model_u_checkpoint.pth")
            # torch.save(model_b.state_dict(), "model_b_checkpoint.pth")
            torch.save(model_f.state_dict(), "model_f_checkpoint.pth")
            torch.save(optimizer.state_dict(), "optimizer_checkpoint.pth")
            with open("checkpoint.txt", "w") as f:
                f.write(str(epoch))

    except KeyboardInterrupt:
        torch.save(model_u.state_dict(), "model_u_checkpoint.pth")
        # torch.save(model_b.state_dict(), "model_b_checkpoint.pth")
        torch.save(model_f.state_dict(), "model_f_checkpoint.pth")
        torch.save(optimizer.state_dict(), "optimizer_checkpoint.pth")
        with open("checkpoint.txt", "w") as f:
            f.write(str(epoch))
        sys.exit()

torch.save(model_u.state_dict(), f"model_u.pth")
# torch.save(model_b.state_dict(), f"model_b.pth")
torch.save(model_f.state_dict(), f"model_f.pth")
# np.savetxt("losses.txt", np.array(losses))

if os.path.exists("checkpoint.txt"):
    os.remove("checkpoint.txt")
    os.remove("model_u_checkpoint.pth")
    # os.remove("model_b_checkpoint.pth")
    os.remove("model_f_checkpoint.pth")
    os.remove("optimizer_checkpoint.pth")
