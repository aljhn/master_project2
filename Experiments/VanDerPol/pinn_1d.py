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


def vanderpol(t, x, args):
    mu = args
    x1, x2 = x 
    # dx1 = mu * (x1 - (x1**3.0) / 3.0 - x2)
    # dx2 = x1 / mu
    dx1 = x2
    dx2 = mu * (1.0 - (x1**2.0)) * x2 - x1
    return jnp.stack((dx1, dx2), axis=0)


def generate_data(function, args, key):
    T0 = 0.0
    T1 = 20.0
    h = 0.01
    t = jnp.arange(T0, T1, h)
    saveat = diffrax.SaveAt(ts=t)
    x0 = jax.random.uniform(key, shape=(2,), minval=-4.0, maxval=4.0)
    term = diffrax.ODETerm(function)
    solver = diffrax.Dopri5()
    solution = diffrax.diffeqsolve(term, solver, t0=T0, t1=T1, dt0=h, y0=x0, saveat=saveat, args=args, adjoint=diffrax.NoAdjoint())
    return t, solution.ys


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
model_def = [1, 50, 50, 50, 1]
params = model_init(model_def, subkey)

optimizer = optax.adamw(learning_rate=5e-3)
opt_state = optimizer.init(params)


def model_forward(x, params):
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


model_x = jax.vmap(model_forward, in_axes=(0, None))
model_dx = jax.vmap(jax.jacfwd(model_forward), in_axes=(0, None))
model_ddx = jax.vmap(jax.hessian(model_forward), in_axes=(0, None))


def pinn(t, params):
    x = model_x(t, params)
    dx = model_dx(t, params)[:, :, 0]
    ddx = model_ddx(t, params)[:, :, 0, 0]
    f = ddx - mu * (1.0 - (x**2.0)) * dx + x
    return f


def u_loss(x_pred, x_true):
    return jnp.mean((x_pred - x_true)**2)


def f_loss(f):
    return jnp.mean(f**2)


def loss(x_pred, x_true, f, beta=0.01):
    return u_loss(x_pred, x_true) + beta * f_loss(f)
    # return beta * f_loss(f)


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


def plot_x(t, x_true):
    x = model_x(t, params)
    plt.figure()
    plt.plot(t, x_true[:, 0])
    plt.plot(t, x)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.legend(["True system", "Learned system"])
    plt.title("Van der Pol")
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
function = vanderpol
mu = 1.0
args = mu
t_batch, x_batch = generate_data(function, args, subkey)
t_batch = jnp.expand_dims(t_batch, axis=1)
t_boundary = jnp.array((t_batch[0],))
# t_boundary = t_batch
t_interior = t_batch[1:, :]
x_boundary = jnp.array(((x_batch[0, 0],),))
# x_boundary = x_batch[:, 0]
# x_boundary = jnp.expand_dims(x_boundary, axis=1)
x_interior = jnp.expand_dims(x_batch[1:, 0], axis=1)
t_pinn = jnp.expand_dims(jnp.linspace(0.0, 20.0, 1000), 1)


epochs = 1000
train_losses = np.zeros(epochs)
val_losses = np.zeros(epochs)
for epoch in range(1, epochs + 1):
    try:
        train_loss, params, opt_state = train(params, opt_state, t_pinn, t_boundary, x_boundary)
        val_loss = validate(params, t_pinn, t_interior, x_interior)
        train_losses[epoch - 1] = train_loss
        val_losses[epoch - 1] = val_loss

        print(f"Epoch: {epoch:3d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    except KeyboardInterrupt:
        break

# plot_loss(train_losses, val_losses)
plot_x(t_batch, x_batch)

