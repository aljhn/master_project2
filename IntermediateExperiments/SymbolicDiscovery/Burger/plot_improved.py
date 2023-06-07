import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


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
        
        self.fourier_omegas = torch.arange(1, m + 1, 1).unsqueeze(0) * 2.0 * np.pi / L


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
model_u = ModifiedMLP(m, L, output_dim, hidden_dim, layers)
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

model_u.load_state_dict(torch.load("model_u.pth"))
model_f.load_state_dict(torch.load("model_f.pth"))
model_true.load_state_dict(torch.load("model_vanilla.pth"))

#with torch.no_grad():
if True:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    n = 100
    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    ttxx.requires_grad = True

    uu_true = model_true(ttxx).squeeze()
    uu_grad = torch.autograd.grad(uu_true, ttxx, grad_outputs=torch.ones_like(uu_true), retain_graph=True, create_graph=True)[0]
    uu_t = uu_grad[:, 0]
    uu_x = uu_grad[:, 1]

    operator_true = - uu_true * uu_x
    uu_collection = torch.stack((uu_true, uu_t, uu_x), dim=1)
    operator_pred = model_f(uu_collection).squeeze()

    op_diff = torch.mean((operator_pred - operator_true)**2)
    print(f"Operator difference: {op_diff:.6f}")
    # Standard: 0.285659
    # ModifiedMLP + Fourier Embedding: 0.194059

    uu = model_u(ttxx).squeeze()
    uu_diff = torch.mean((uu - uu_true)**2)
    print(f"U difference: {uu_diff:.6f}")
    # Standard: 0.000128
    # ModifiedMLP + Fourier Embedding: 0.000077

    un = torch.reshape(uu, tt.shape).detach()

    plt.figure()
    plt.pcolormesh(t, x, un, vmin=-1.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("burger.pdf")
    plt.show()
