"""Tests for DDPM ancestral reverse sampling."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ddpm.diffusion import make_linear_schedule
from ddpm.model import UNet
from ddpm.sampling import p_sample_step, reverse_mean, sample


def zero_predictor(variables, noisy_images, timesteps):
    del variables, timesteps
    return jnp.zeros_like(noisy_images)


class SamplingTests(unittest.TestCase):
    def test_reverse_mean_matches_ddpm_equation(self):
        schedule = make_linear_schedule(4, beta_start=0.01, beta_end=0.04)
        noisy_images = jnp.full((2, 3, 3, 1), 0.5)
        predicted_noise = jnp.full_like(noisy_images, -0.25)
        timestep = 2

        actual = reverse_mean(
            schedule, noisy_images, predicted_noise, timestep
        )
        expected = (
            noisy_images
            - schedule.betas[timestep]
            * predicted_noise
            / jnp.sqrt(1.0 - schedule.alpha_bars[timestep])
        ) / jnp.sqrt(schedule.alphas[timestep])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_final_step_is_deterministic(self):
        schedule = make_linear_schedule(10)
        noisy_images = jnp.ones((2, 4, 4, 1))
        first = p_sample_step(
            zero_predictor,
            {},
            schedule,
            noisy_images,
            jnp.array(0),
            jax.random.key(0),
        )
        second = p_sample_step(
            zero_predictor,
            {},
            schedule,
            noisy_images,
            jnp.array(0),
            jax.random.key(1),
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(
            first,
            reverse_mean(schedule, noisy_images, jnp.zeros_like(noisy_images), 0),
        )

    def test_nonfinal_step_uses_posterior_variance(self):
        schedule = make_linear_schedule(10)
        noisy_images = jnp.ones((2, 4, 4, 1))
        timestep = 5
        key = jax.random.key(0)
        actual = p_sample_step(
            zero_predictor,
            {},
            schedule,
            noisy_images,
            jnp.array(timestep),
            key,
        )
        posterior_variance = (
            schedule.betas[timestep]
            * (1.0 - schedule.alpha_bars[timestep - 1])
            / (1.0 - schedule.alpha_bars[timestep])
        )
        expected = reverse_mean(
            schedule, noisy_images, jnp.zeros_like(noisy_images), timestep
        ) + jnp.sqrt(posterior_variance) * jax.random.normal(
            key, noisy_images.shape
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_full_sampler_with_unet(self):
        schedule = make_linear_schedule(4)
        model = UNet(base_channels=8)
        example_images = jnp.zeros((2, 28, 28, 1), dtype=jnp.float32)
        timesteps = jnp.zeros((2,), dtype=jnp.int32)
        params = model.init(
            jax.random.key(0), example_images, timesteps
        )["params"]

        generated = sample(
            model.apply, params, schedule, jax.random.key(1), num_samples=2
        )
        repeated = sample(
            model.apply, params, schedule, jax.random.key(1), num_samples=2
        )
        self.assertEqual(generated.shape, (2, 28, 28, 1))
        self.assertEqual(generated.dtype, jnp.float32)
        self.assertEqual(generated.devices(), example_images.devices())
        self.assertTrue(np.isfinite(np.asarray(generated)).all())
        np.testing.assert_array_equal(generated, repeated)

    def test_conditional_sampler_is_reproducible(self):
        schedule = make_linear_schedule(4)
        model = UNet(base_channels=8)
        images = jnp.zeros((2, 28, 28, 1), dtype=jnp.float32)
        timesteps = jnp.zeros((2,), dtype=jnp.int32)
        covariates = jax.nn.one_hot(jnp.array([3, 7]), 10)
        params = model.init(
            jax.random.key(0), images, timesteps, covariates
        )["params"]
        generated = sample(
            model.apply,
            params,
            schedule,
            jax.random.key(1),
            num_samples=2,
            covariates=covariates,
        )
        repeated = sample(
            model.apply,
            params,
            schedule,
            jax.random.key(1),
            num_samples=2,
            covariates=covariates,
        )
        self.assertEqual(generated.shape, images.shape)
        self.assertTrue(np.isfinite(np.asarray(generated)).all())
        np.testing.assert_array_equal(generated, repeated)


if __name__ == "__main__":
    unittest.main()
