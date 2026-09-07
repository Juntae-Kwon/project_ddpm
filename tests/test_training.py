"""Tests for the DDPM objective and optimizer update."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ddpm.diffusion import make_linear_schedule
from ddpm.model import UNet
from ddpm.training import create_train_state, noise_prediction_loss, train_step


class TrainingTests(unittest.TestCase):
    def test_exact_noise_prediction_has_zero_loss(self):
        schedule = make_linear_schedule(10)
        clean_images = jnp.ones((2, 4, 4, 1), dtype=jnp.float32)
        key = jax.random.key(3)
        _, noise_key = jax.random.split(key)
        exact_noise = jax.random.normal(noise_key, clean_images.shape)

        covariates = jax.nn.one_hot(jnp.array([2, 8]), 10)

        def exact_predictor(variables, noisy_images, timesteps, labels):
            del variables, noisy_images, timesteps
            self.assertEqual(labels.shape, (2, 10))
            return exact_noise

        loss = noise_prediction_loss(
            {}, exact_predictor, schedule, clean_images, key, covariates
        )
        self.assertEqual(float(loss), 0.0)

    def test_train_step_updates_model_and_reduces_fixed_batch_loss(self):
        model = UNet(base_channels=8)
        clean_images = jax.random.uniform(
            jax.random.key(0),
            (2, 28, 28, 1),
            minval=-1.0,
            maxval=1.0,
        )
        schedule = make_linear_schedule(100)
        state = create_train_state(
            model, jax.random.key(1), clean_images, learning_rate=1e-3
        )
        initial_params = state.params
        training_key = jax.random.key(2)
        initial_loss = noise_prediction_loss(
            state.params, state.apply_fn, schedule, clean_images, training_key
        )

        compiled_step = jax.jit(train_step)
        losses = []
        for _ in range(4):
            state, loss = compiled_step(
                state, schedule, clean_images, training_key
            )
            losses.append(float(loss))
        final_loss = noise_prediction_loss(
            state.params, state.apply_fn, schedule, clean_images, training_key
        )

        self.assertEqual(int(state.step), 4)
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(np.isfinite(np.asarray(final_loss)))
        self.assertLess(float(final_loss), float(initial_loss))
        changed = [
            not np.array_equal(before, after)
            for before, after in zip(
                jax.tree.leaves(initial_params), jax.tree.leaves(state.params)
            )
        ]
        self.assertTrue(any(changed))
        for value in jax.tree.leaves((state.params, state.opt_state, final_loss)):
            if hasattr(value, "devices"):
                self.assertEqual(value.devices(), clean_images.devices())

    def test_learning_rate_must_be_positive(self):
        with self.assertRaises(ValueError):
            create_train_state(
                UNet(base_channels=8),
                jax.random.key(0),
                jnp.zeros((1, 28, 28, 1)),
                learning_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
