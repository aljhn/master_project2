import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


n = 100

T0 = 0.0
T1 = 5.0
X0 = 0.0
X1 = 4.0

nu = 0.01

output_dim = 1
hidden_dim = 50
layers = 5
L = X1 - X0
m = 5
input_dim = 2 * m + 1 + 1


def solution(t, x):
    return (
        2.0
        * nu
        * np.pi
        * torch.exp(-(np.pi**2.0) * nu * (t - 5.0))
        * torch.sin(np.pi * x)
        / (2.0 + torch.exp(-(np.pi**2.0) * nu * (t - 5.0)) * torch.cos(np.pi * x))
    )


fourier_omegas = torch.arange(1, m + 1, 1).unsqueeze(0) * 2.0 * np.pi / L


def fourier_embedding(x):
    embedding = torch.cat(
        (
            torch.ones_like(x),
            torch.cos(x @ fourier_omegas),
            torch.sin(x @ fourier_omegas),
        ),
        dim=1,
    )
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
    model_u = ModifiedMLP(input_dim, output_dim, hidden_dim, layers)
    model_u.load_state_dict(torch.load("model_u.pth", map_location=torch.device("cpu")))

    t = torch.linspace(T0, T1, n)
    x = torch.linspace(X0, X1, n)
    tt, xx = torch.meshgrid(t, x, indexing="xy")
    ttxx = torch.stack((tt.flatten(), xx.flatten()), dim=1)
    uu = model_u(ttxx)
    un = torch.reshape(uu, tt.shape)

    plt.figure()
    plt.pcolormesh(t.cpu(), x.cpu(), un.cpu(), vmin=-1.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("burger.pdf")
    plt.show()

    u_true = solution(tt, xx)

    plt.figure()
    plt.plot(x, u_true[:, 0])
    plt.plot(x, un[:, 0])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.legend(["True Solution", "Learned Solution"])
    plt.ylim([-0.1, 0.1])
    plt.tight_layout()
    plt.savefig("burger_slice0.pdf")
    plt.show()

    plt.figure()
    plt.plot(x, u_true[:, -1])
    plt.plot(x, un[:, -1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.legend(["True Solution", "Learned Solution"])
    plt.ylim([-0.1, 0.1])
    plt.tight_layout()
    plt.savefig("burger_slice1.pdf")
    plt.show()
