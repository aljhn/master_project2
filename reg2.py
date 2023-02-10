import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import equinox as eqx


seed = 42069
key = jax.random.PRNGKey(seed)


key, subkey = jax.random.split(key)
# model = eqx.nn.MLP(2, 1, 100, 3, activation=jnp.tanh, key=subkey)

# optimizer = optax.adam(learning_rate=3e-3)
# opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


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


model_def = [2, 100, 100, 100, 1]
params = model_init(model_def, subkey)

optimizer = optax.adam(learning_rate=3e-3)
opt_state = optimizer.init(params)


def model_forward(t, x, params):
    x = jnp.concatenate((t, x), axis=0)
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


u = jax.vmap(model_forward, in_axes=(0, 0, None))


@eqx.filter_value_and_grad
def train_step(params, t, x, u_true):
    # u_pred = jax.vmap(model)(x)
    u_pred = u(t, x, params)
    return jnp.mean((u_true - u_pred)**2)


@eqx.filter_jit
def train(params, opt_state, t, x, u_true):
    loss_value, loss_grads = train_step(params, t, x, u_true)
    updates, opt_state = optimizer.update(loss_grads, opt_state, params)
    # model = eqx.apply_updates(model, updates)
    params = optax.apply_updates(params, updates)
    return loss_value, params, opt_state


t = jnp.zeros((100, 1))
x = jnp.expand_dims(jnp.linspace(0, 1, 100), axis=1)
u_true = jnp.sin(jnp.pi * x)
# tx = jnp.concatenate((t, x), axis=1)

epochs = 1000
for epoch in range(1, epochs + 1):
    try:
        train_loss, params, opt_state = train(params, opt_state, t, x, u_true)
        print(f"Epoch: {epoch:3d}, Train Loss: {train_loss.item():.5f}")
    except KeyboardInterrupt:
        break


plt.figure()
plt.scatter(x, u_true)
# uu = jax.vmap(model)(tx)
uu = u(t, x, params)
plt.scatter(x, uu)
plt.show()
