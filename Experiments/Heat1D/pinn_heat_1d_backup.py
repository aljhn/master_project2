import random
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

seed = 42069
random.seed(seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)


def heat(t, x):
    return jnp.sin(jnp.pi * x) * jnp.exp(-jnp.pi**2 * t)


def generate_data(function, T0, T1, X0, X1, batch_size, key):
    key1, key2, key3, key4 = jax.random.split(key, 4)

    t_initial = jnp.ones((batch_size // 2, 1)) * T0
    x_initial = jax.random.uniform(key1, shape=(batch_size // 2, 1), minval=X0, maxval=X1)
    u_initial = function(T0, x_initial)

    t_boundary = jax.random.uniform(key2, shape=(batch_size // 2, 1), minval=T0, maxval=T1)
    x_boundary0 = jnp.ones((batch_size // 4, 1)) * X0
    x_boundary1 = jnp.ones((batch_size // 4, 1)) * X1
    x_boundary = jnp.concatenate((x_boundary0, x_boundary1), axis=0)
    u_boundary = function(t_boundary, x_boundary)

    t_ib = jnp.concatenate((t_initial, t_boundary), axis=0)
    x_ib = jnp.concatenate((x_initial, x_boundary), axis=0)
    u_ib = jnp.concatenate((u_initial, u_boundary), axis=0)

    t_interior = jax.random.uniform(key3, shape=(batch_size, 1), minval=T0, maxval=T1)
    x_interior = jax.random.uniform(key4, shape=(batch_size, 1), minval=X0, maxval=X1)
    u_interior = function(t_interior, x_interior)

    return t_ib, x_ib, u_ib, t_interior, x_interior, u_interior


def data_split(x, ratio=0.8):
    batch_size = x.shape[0]
    x_train = x[:int(batch_size * ratio), :]
    x_val = x[int(batch_size * ratio):, :]
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
model_def = [2, 100, 100, 100, 1]
params = model_init(model_def, subkey)

optimizer = optax.adamw(learning_rate=1e-3)
opt_state = optimizer.init(params)


def model_forward(t, x, params):
    x = jnp.concatenate((t, x), axis=0)
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jax.nn.relu(x)
            # x = jnp.tanh(x)
    return x[0]


u = jax.vmap(model_forward, in_axes=(0, 0, None))
u_t = jax.vmap(jax.grad(model_forward, argnums=0), in_axes=(0, 0, None))
u_xx = jax.vmap(jax.jacfwd(jax.jacfwd(model_forward, argnums=1), argnums=1), in_axes=(0, 0, None))


def pinn(t, x, params):
    f1 = u_t(t, x, params).squeeze()
    f2 = u_xx(t, x, params).squeeze()
    f = f1 - f2
    return f


def u_loss(u_pred, u_true):
    return jnp.mean((u_pred - u_true)**2)


def f_loss(f):
    return jnp.mean(f**2)


def loss(u_pred, u_true, f, beta=1.0):
    return u_loss(u_pred, u_true)# + beta * f_loss(f)


@jax.value_and_grad
def train_step(params, t_ib, x_ib, u_ib, t_f, x_f):
    u_pred = u(t_ib, x_ib, params)
    f = pinn(t_f, x_f, params)
    return loss(u_pred, u_ib, f)


@jax.jit
def train(params, opt_state, t_ib, x_ib, u_ib, t_f, x_f):
    loss_value, loss_grads = train_step(params, t_ib, x_ib, u_ib, t_f, x_f)
    updates, opt_state = optimizer.update(loss_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss_value, params, opt_state


@jax.jit
def validate(params, t_interior, x_interior, u_interior, t_f, x_f):
    u_pred = u(t_interior, x_interior, params)
    f = pinn(t_f, x_f, params)
    return loss(u_pred, u_interior, f)


def plot_u():
    n = 100
    t = np.linspace(T0, T1, n)
    x = np.linspace(X0, X1, n)
    tt = np.zeros((n * n, 1))
    xx = np.zeros((n * n, 1))
    for i in range(n):
        for j in range(n):
            tt[i * n + j, 0] = t[i]
            xx[i * n + j, 0] = x[j]
    uu = u(tt, xx, params)
    uu_true = heat(tt, xx)
    print(jnp.mean((uu - uu_true)**2))
    # exit()
    un = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            un[j, i] = uu[i * n + j]
            # un[j, i] = uu_true[i * n + j]
    plt.figure()
    plt.pcolormesh(t, x, un)
    plt.colorbar()
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


function = heat
T0 = 0.0
T1 = 0.2
X0 = 0.0
X1 = 1.0
batch_size = 100
key, subkey = jax.random.split(key)
t_ib, x_ib, u_ib, t_interior, x_interior, u_interior = generate_data(function, T0, T1, X0, X1, batch_size, subkey)
n_f = 100
t_f = jnp.expand_dims(jnp.linspace(T0, T1, n_f), 1)
x_f = jnp.expand_dims(jnp.linspace(X0, X1, n_f), 1)

t_i = t_ib[:batch_size // 2, :]
x_i = x_ib[:batch_size // 2, :]
u_i = u_ib[:batch_size // 2, :]

t_b = t_ib[batch_size // 2:, :]
x_b = x_ib[batch_size // 2:, :]
u_b = u_ib[batch_size // 2:, :]

epochs = 10000
train_losses = np.zeros(epochs)
val_losses = np.zeros(epochs)
for epoch in range(1, epochs + 1):
    try:
        t_train_i = jnp.expand_dims(t_i[epoch % (batch_size // 2), :], axis=0)
        x_train_i = jnp.expand_dims(x_i[epoch % (batch_size // 2), :], axis=0)
        u_train_i = jnp.expand_dims(u_i[epoch % (batch_size // 2), :], axis=0)

        t_train_b = jnp.expand_dims(t_b[epoch % (batch_size // 2), :], axis=0)
        x_train_b = jnp.expand_dims(x_b[epoch % (batch_size // 2), :], axis=0)
        u_train_b = jnp.expand_dims(u_b[epoch % (batch_size // 2), :], axis=0)

        t_train = jnp.concatenate((t_train_i, t_train_b), axis=0)
        x_train = jnp.concatenate((x_train_i, x_train_b), axis=0)
        u_train = jnp.concatenate((u_train_i, u_train_b), axis=0)

        train_loss, params, opt_state = train(params, opt_state, t_i, x_i, u_i, t_f, x_f)
        val_loss = validate(params, t_interior, x_interior, u_interior, t_f, x_f)
        train_losses[epoch - 1] = train_loss
        val_losses[epoch - 1] = val_loss

        print(f"Epoch: {epoch:3d}, Train Loss: {train_loss.item():.5f}, Val Loss: {val_loss.item():.5f}")
    except KeyboardInterrupt:
        break

plt.figure()
plt.scatter(x_i, u_i)
uu_i = u(t_i, x_i, params)
plt.scatter(x_i, uu_i)
plt.show()
exit()

# plot_loss(train_losses, val_losses)
plot_u()

