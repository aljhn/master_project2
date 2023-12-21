import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

val_losses = np.loadtxt("val_losses_lessdata")
val_losses_maxmin = np.loadtxt("val_losses_maxmin_lessdata")

epochs = val_losses.shape[0]

plt.figure()
plt.plot(np.arange(1, epochs + 1), val_losses)
plt.plot(np.arange(1, epochs + 1), val_losses_maxmin)
plt.yscale("log")
plt.xlabel(r"Epochs")
plt.ylabel(r"Loss")
plt.legend(["Standard", "Regularized"])
plt.tight_layout()
plt.savefig("maxmin_losses_lessdata.pdf")
plt.show()
