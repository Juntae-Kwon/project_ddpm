"""Tests for the timestep-conditioned MNIST denoising network."""

import unittest

from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from ddpm.model import UNet, sinusoidal_timestep_embedding


class ModelTests(unittest.TestCase):
    def test_timestep_embedding(self):
        embedding = sinusoidal_timestep_embedding(jnp.array([0, 1]), 8)
        self.assertEqual(embedding.shape, (2, 8))
        np.testing.assert_array_equal(embedding[0, :4], 0.0)
        np.testing.assert_array_equal(embedding[0, 4:], 1.0)
        self.assertFalse(np.array_equal(embedding[0], embedding[1]))

        for invalid_dim in (2, 5):
            with self.subTest(invalid_dim=invalid_dim), self.assertRaises(ValueError):
                sinusoidal_timestep_embedding(jnp.array([0]), invalid_dim)

    def test_unet_shape_conditioning_gradients_and_device(self):
        model = UNet(base_channels=8)
        images = jnp.ones((2, 28, 28, 1), dtype=jnp.float32)
        timesteps = jnp.array([0, 999], dtype=jnp.int32)
        variables = model.init(jax.random.key(0), images, timesteps)

        apply_model = jax.jit(model.apply)
        predictions = apply_model(variables, images, timesteps)
        self.assertEqual(predictions.shape, images.shape)
        self.assertEqual(predictions.dtype, images.dtype)
        self.assertEqual(predictions.devices(), images.devices())
        self.assertTrue(np.isfinite(np.asarray(predictions)).all())
        self.assertFalse(np.allclose(predictions[0], predictions[1]))

        def mean_prediction(params):
            output = model.apply({"params": params}, images, timesteps)
            return jnp.mean(output**2)

        loss, gradients = jax.jit(jax.value_and_grad(mean_prediction))(
            variables["params"]
        )
        self.assertTrue(np.isfinite(np.asarray(loss)))
        for value in jax.tree.leaves(gradients):
            self.assertTrue(np.isfinite(np.asarray(value)).all())
            self.assertEqual(value.devices(), images.devices())

        parameter_count = sum(
            value.size for value in flatten_dict(variables["params"]).values()
        )
        self.assertGreater(parameter_count, 0)
        self.assertLess(parameter_count, 1_000_000)

    def test_base_channels_validation(self):
        images = jnp.zeros((1, 28, 28, 1))
        timesteps = jnp.zeros((1,), dtype=jnp.int32)
        for channels in (4, 10):
            with self.subTest(channels=channels), self.assertRaises(ValueError):
                UNet(base_channels=channels).init(
                    jax.random.key(0), images, timesteps
                )


if __name__ == "__main__":
    unittest.main()
