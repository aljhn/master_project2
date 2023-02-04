import random
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import diffrax


seed = 42069
random.seed(seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)


def riccati(t, x, args):
    return x * x - t


def generate_data(function, args, key):
    T0 = 0.0
    T1 = 10.0
    h = 0.01
    t = jnp.arange(T0, T1 + h, h)
    saveat = diffrax.SaveAt(ts=t)
    x0 = jax.random.uniform(key, minval=-5.0, maxval=0.0)
    term = diffrax.ODETerm(function)
    solver = diffrax.Dopri5()
    solution = diffrax.diffeqsolve(term, solver, t0=T0, t1=T1, dt0=h, y0=x0, saveat=saveat, args=args, adjoint=diffrax.NoAdjoint())
    return t, solution.ys


def data_split(x, ratio=0.8):
    batch_size = x.shape[0]
    x_train = x[:int(batch_size * ratio)]
    x_val = x[int(batch_size * ratio):]
    return x_train, x_val


def model_init(model_def, key):
    subkeys = jax.random.split(key, num=(len(model_def) - 1) * 2)
    params = []
    for i in range(len(model_def) - 1):
        layer = {
            "weights": jax.random.normal(subkeys[i], (model_def[i], model_def[i + 1])),
            "bias": jax.random.normal(subkeys[i + len(model_def) - 1], (model_def[i + 1],))
        }
        params.append(layer)
    return params


key, subkey = jax.random.split(key)
model_def = [1, 50, 100, 100, 50, 1]
params = model_init(model_def, subkey)


optimizer = optax.adamw(learning_rate=1e-3)
opt_state = optimizer.init(params)


def model_forward(x, params):
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jax.nn.sigmoid(x)
    return x


model_x = jax.vmap(model_forward, in_axes=(0, None))
model_x_t = jax.vmap(jax.jacfwd(model_forward), in_axes=(0, None))


def pinn(t, params):
    f1 = model_x_t(t, params)
    f2 = model_x(t, params)**2 - t
    f = f1 - f2
    return f


def u_loss(x_pred, x_true):
    return jnp.mean((x_pred - x_true)**2)


def f_loss(f):
    return jnp.mean(f**2)


def loss(x_pred, x_true, f, beta=0.01):
    return u_loss(x_pred, x_true) + beta * f_loss(f)


@jax.value_and_grad
def train_step(params, t, t_boundary, x_boundary):
    x_pred = model_x(t_boundary, params)
    f = pinn(t, params)
    return loss(x_pred, x_boundary, f)


@jax.jit
def train(params, opt_state, t, t_boundary, x_boundary):
    loss_value, loss_grads = train_step(params, t, t_boundary, x_boundary)
    updates, opt_state = optimizer.update(loss_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss_value, params, opt_state


@jax.jit
def validate(params, t, t_interior, x_interior):
    x_pred = model_x(t_interior, params)
    f = pinn(t, params)
    return loss(x_pred, x_interior, f)


def plot_x():
    n = 100
    t = np.linspace(0, 10, n)
    t = np.expand_dims(t, 1)
    x = model_x(t, params)
    plt.figure()
    plt.plot(t_batch, x_batch)
    plt.plot(t, x)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["True system", "Learned system"])
    plt.show()


def plot_loss(train_losses, val_losses):
    train_losses = train_losses[train_losses > 0]
    val_losses = val_losses[val_losses > 0]
    epochs = max(train_losses.shape[0], val_losses.shape[0])
    plt.figure()
    plt.plot(np.arange(1, epochs + 1, 1), train_losses)
    plt.plot(np.arange(1, epochs + 1, 1), val_losses)
    plt.legend(["Training loss", "Validation loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("")
    plt.yscale("log")
    plt.grid()
    plt.show()


key, subkey = jax.random.split(key)
function = riccati
args = None
t_batch, x_batch = generate_data(function, None, subkey)
t_batch = jnp.expand_dims(t_batch, 1)
x_batch = jnp.expand_dims(x_batch, 1)
t_boundary = jnp.stack((t_batch[0], t_batch[-1]), axis=0)
t_interior = t_batch[1:-1, :]
x_boundary = jnp.stack((x_batch[0], x_batch[-1]), axis=0)
x_interior = x_batch[1:-1, :]

epochs = 1000
train_losses = np.zeros(epochs)
val_losses = np.zeros(epochs)
for epoch in range(1, epochs + 1):
    try:
        train_loss, params, opt_state = train(params, opt_state, t_batch, t_boundary, x_boundary)
        val_loss = validate(params, t_batch, t_interior, x_interior)
        train_losses[epoch - 1] = train_loss
        val_losses[epoch - 1] = val_loss

        print(f"Epoch: {epoch:3d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    except KeyboardInterrupt:
        break

# plot_loss(train_losses, val_losses)
plot_x()

