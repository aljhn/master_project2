import random
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

seed = 42069
random.seed(seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)


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
model_def = [1, 100, 100, 100, 1]
params = model_init(model_def, subkey)

optimizer = optax.adamw(learning_rate=1e-3)
opt_state = optimizer.init(params)


def model_forward(x, params):
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            # x = jax.nn.relu(x)
            x = jnp.tanh(x)
    return x[0]


u = jax.vmap(model_forward, in_axes=(0, None))


key, subkey = jax.random.split(key)
mlp = eqx.nn.MLP(1, 1, 100, 3, activation=jnp.tanh, key=subkey)

# optimizer = optax.adamw(learning_rate=1e-3)
# opt_state = optimizer.init(eqx.filter(mlp, eqx.is_inexact_array))


def loss(u_pred, u_true):
    return jnp.mean(optax.l2_loss(u_pred, u_true))


@jax.value_and_grad
# @eqx.filter_value_and_grad
def train_step(params, x_ib, u_ib):
    u_pred = u(x_ib, params)
    # u_pred = params(x_ib.T).squeeze()
    return loss(u_pred, u_ib)


@jax.jit
# @eqx.filter_jit
def train(params, opt_state, x_ib, u_ib):
    loss_value, loss_grads = train_step(params, x_ib, u_ib)
    updates, opt_state = optimizer.update(loss_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # params = eqx.apply_updates(params, updates)
    return loss_value, params, opt_state


x_i = jnp.expand_dims(jnp.linspace(0.0, 1.0, 100), 1)
u_i = jnp.sin(jnp.pi * x_i).squeeze()
# u_i = 2.0 * x_i.squeeze()

epochs = 10000
for epoch in range(1, epochs + 1):
    try:
        # train_loss, mlp, opt_state = train(mlp, opt_state, x_i, u_i)
        train_loss, params, opt_state = train(params, opt_state, x_i, u_i)
        print(f"Epoch: {epoch:3d}, Train Loss: {train_loss.item():.5f}")
    except KeyboardInterrupt:
        break

plt.figure()
plt.scatter(x_i, u_i)
uu_i = u(x_i, params)
# uu_i = mlp(x_i.T)
plt.scatter(x_i, uu_i)
plt.show()

