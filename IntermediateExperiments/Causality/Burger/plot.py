import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


n = 100
t_iterations = 10

T0 = 0.0
T1 = 1.0
X0 = -1.0
X1 = 1.0

with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    t = torch.linspace(T0, T1, n * t_iterations)
    x = torch.linspace(X0, X1, n)
    un = torch.zeros((n, n * t_iterations))
    for i in range(n * t_iterations):
        try:
            un[:, n * i : n * (i + 1)] = torch.tensor(np.loadtxt(f"u_{i}.txt"))
        except:
            break
    # print(un.shape)
    # exit()
    plt.figure()
    plt.pcolormesh(t.cpu(), x.cpu(), un.cpu(), vmin=-1.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.savefig("burger.pdf")
    plt.show()

    n = un.shape[1]

    plt.figure()
    plt.plot(x, un[:, 0])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice1.pdf")
    plt.show()

    plt.figure()
    plt.plot(x, un[:, n // 3 - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice2.pdf")
    plt.show()

    plt.figure()
    plt.plot(x, un[:, n * 2 // 3 - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice3.pdf")
    plt.show()

    plt.figure()
    plt.plot(x, un[:, n - 1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(t, x)$")
    plt.xlim([X0, X1])
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("burger_slice4.pdf")
    plt.show()
