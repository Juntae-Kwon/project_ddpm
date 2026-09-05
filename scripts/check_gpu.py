"""Fail unless JAX executes a small compiled calculation on an NVIDIA GPU."""

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


def main():
    print(f"JAX: {jax.__version__}; jaxlib: {jaxlib.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("JAX is not using its CUDA GPU backend")

    device = jax.devices("gpu")[0]
    values = jax.device_put(jnp.arange(16, dtype=jnp.float32), device)
    result = jax.jit(lambda x: jnp.sum(x * x))(values)
    assert result.devices() == {device}
    np.testing.assert_allclose(result, 1240.0)
    print(f"Compiled calculation device: {result.devices()}")
    print(f"Device kind: {device.device_kind}")
    print(f"Memory statistics: {device.memory_stats()}")
    print("CUDA-backed JAX verification passed.")


if __name__ == "__main__":
    main()
