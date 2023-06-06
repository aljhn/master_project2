import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


n = 100
t_iterations = 1

T0 = 0.0
T1 = 1.0
X0 = -1.0
X1 = 1.0

output_dim = 1
hidden_dim = 100
layers = 4
L = X1 - X0
m = 10
input_dim = 2 * m + 1 + 1

fourier_omegas = torch.arange(1, m + 1, 1).unsqueeze(0) * 2.0 * np.pi / L

def fourier_embedding(x):
    embedding = torch.cat((torch.ones_like(x), torch.cos(x @ fourier_omegas), torch.sin(x @ fourier_omegas)), dim=1)
    return embedding

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


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

with torch.no_grad():
    """t = torch.linspace(T0, T1, n * t_iterations)
    x = torch.linspace(X0, X1, n)
    un = torch.zeros((n, n * t_iterations))
    for i in range(n * t_iterations):
        try:
            un[:, n * i:n * (i + 1)] = torch.tensor(np.loadtxt(f"u_{i}.txt"))
        except:
            break"""

    model = ModifiedMLP(input_dim, output_dim, hidden_dim, layers)
    model.load_state_dict(torch.load("model_checkpoint.pth"))

    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    uu = model(ttxx)
    un = torch.reshape(uu, tt.shape)

    plt.figure()
    plt.pcolormesh(t.cpu(), x.cpu(), un.cpu(), vmin=-1.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("ac.pdf")
    plt.show()
