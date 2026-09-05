"""Check the installed numerical stack using a tiny device computation."""

import sys

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    print(f"Python: {sys.version.split()[0]}")
    for package in (jax, flax, optax, np):
        print(f"{package.__name__}: {package.__version__}")
    print(f"JAX devices: {jax.devices()}")

    # No backend override: JAX chooses its default available device.
    x = jax.device_put(np.ones((2, 3), dtype=np.float32))
    model = nn.Dense(features=1)
    params = model.init(jax.random.key(0), x)["params"]

    def loss_fn(params):
        return jnp.mean(model.apply({"params": params}, x) ** 2)

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params)
    optimizer = optax.sgd(learning_rate=0.01)
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    params = optax.apply_updates(params, updates)
    output = model.apply({"params": params}, x)
    assert output.shape == (2, 1)
    for value in jax.tree.leaves((loss, grads, params, output)):
        assert np.isfinite(np.asarray(value)).all()
        assert value.devices() == x.devices()
    print(f"Tensor device: {x.devices()}; loss: {float(loss):.6f}")
    print("Environment check passed.")


if __name__ == "__main__":
    main()
