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


def heat(t, x, k=1):
    return 1 / np.sqrt(4 * np.pi * k * t) * np.exp(-x**2 / (4 * k * t))


def generate_data(function, params, batch_size, key):
    key1, key2 = jax.random.split(key)
    t = jax.random.uniform(key2, shape=(batch_size, 1), minval=1, maxval=10)
    x = jax.random.uniform(key1, shape=(batch_size, 1), minval=-10, maxval=10)
    u = function(t, x, params)
    return t, x, u


def data_split(x, ratio=0.8):
    batch_size = x.shape[0]
    x_train = x[:int(batch_size * ratio), :]
    x_val = x[int(batch_size * ratio):, :]
    return x_train, x_val


batch_size = 200
function = heat
params = 1
key, subkey = jax.random.split(key)
t, x, u = generate_data(function, params, batch_size, subkey)
t_train, t_val = data_split(t)
x_train, x_val = data_split(x)
u_train, u_val = data_split(u)


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
model_def = [2, 50, 100, 50, 1]
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
            x = jax.nn.sigmoid(x)
    return x[0]


u = jax.vmap(model_forward, in_axes=(0, 0, None))
u_t = jax.vmap(jax.grad(model_forward, argnums=0), in_axes=(0, 0, None))
u_xx = jax.vmap(jax.jacfwd(jax.jacrev(model_forward, argnums=1), argnums=1), in_axes=(0, 0, None))


def pinn(t, x, params):
    f1 = u_t(t, x, params).squeeze()
    f2 = u_xx(t, x, params).squeeze()
    f = f1 - f2
    return f


def u_loss(u_pred, u_true):
    return jnp.mean((u_pred - u_true)**2)


def f_loss(f):
    return jnp.mean(f**2)


def loss(u_pred, u_true, f, beta=1):
    return u_loss(u_pred, u_true) + beta * f_loss(f)


@jax.value_and_grad
def train_step(params, t, x, u_true):
    u_pred = u(t, x, params)
    f = pinn(t, x, params)
    return loss(u_pred, u_true, f)


@jax.jit
def train(params, opt_state, t, x, u_true):
    loss_value, loss_grads = train_step(params, t, x, u_true)
    updates, opt_state = optimizer.update(loss_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss_value, params, opt_state


@jax.jit
def validate(params, t, x, u_true):
    u_pred = u(t, x, params)
    f = pinn(t, x, params)
    return loss(u_pred, u_true, f)


epochs = 100
for epoch in range(1, epochs + 1):
    try:
        train_loss, params, opt_state = train(params, opt_state, t_train, x_train, u_train)
        val_loss = validate(params, t_val, x_val, u_val)

        print(f"Epoch: {epoch:3d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    except KeyboardInterrupt:
        break
