"""Tests for the DDPM forward process and its schedule."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ddpm.diffusion import make_linear_schedule, q_sample, sample_forward


class DiffusionTests(unittest.TestCase):
    def test_linear_schedule(self):
        schedule = make_linear_schedule(5, beta_start=0.01, beta_end=0.05)
        np.testing.assert_allclose(schedule.betas, [0.01, 0.02, 0.03, 0.04, 0.05])
        np.testing.assert_allclose(schedule.alphas, 1.0 - schedule.betas)
        np.testing.assert_allclose(schedule.alpha_bars, np.cumprod(schedule.alphas))
        self.assertTrue(np.all(np.diff(np.asarray(schedule.alpha_bars)) < 0))
        for value in schedule:
            self.assertEqual(value.dtype, jnp.float32)
            self.assertEqual(value.devices(), jax.device_put(0).devices())
            self.assertTrue(np.isfinite(np.asarray(value)).all())

    def test_invalid_schedule(self):
        for arguments in (
            (0, 1e-4, 2e-2),
            (10, 0.0, 2e-2),
            (10, 0.02, 0.01),
            (10, 1e-4, 1.0),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                make_linear_schedule(*arguments)

    def test_q_sample_matches_closed_form_for_each_image(self):
        schedule = make_linear_schedule(4, beta_start=0.01, beta_end=0.04)
        clean = jnp.arange(3 * 2 * 2, dtype=jnp.float32).reshape(3, 2, 2, 1)
        noise = jnp.full_like(clean, 0.25)
        timesteps = jnp.array([0, 2, 3])

        noisy = jax.jit(q_sample)(schedule, clean, timesteps, noise)
        expected = np.empty_like(clean)
        for image, timestep in enumerate(np.asarray(timesteps)):
            expected[image] = (
                schedule.sqrt_alpha_bars[timestep] * clean[image]
                + schedule.sqrt_one_minus_alpha_bars[timestep] * noise[image]
            )
        np.testing.assert_allclose(noisy, expected, rtol=1e-6)
        self.assertEqual(noisy.shape, clean.shape)
        self.assertEqual(noisy.devices(), clean.devices())

    def test_noise_contribution_grows_with_timestep(self):
        schedule = make_linear_schedule(100)
        clean = jnp.zeros((2, 4, 4, 1), dtype=jnp.float32)
        noise = jnp.ones_like(clean)
        noisy = q_sample(schedule, clean, jnp.array([0, 99]), noise)
        self.assertGreater(float(noisy[1, 0, 0, 0]), float(noisy[0, 0, 0, 0]))

    def test_sample_forward_is_reproducible_and_finite(self):
        schedule = make_linear_schedule(10)
        clean = jnp.ones((2, 4, 4, 1), dtype=jnp.float32)
        timesteps = jnp.array([1, 8])
        key = jax.random.key(7)
        noisy, noise = jax.jit(sample_forward)(schedule, clean, timesteps, key)
        repeated_noisy, repeated_noise = sample_forward(schedule, clean, timesteps, key)
        np.testing.assert_allclose(noisy, repeated_noisy, rtol=1e-6)
        np.testing.assert_array_equal(noise, repeated_noise)
        np.testing.assert_allclose(noisy, q_sample(schedule, clean, timesteps, noise))
        self.assertTrue(np.isfinite(np.asarray(noisy)).all())
        self.assertTrue(np.isfinite(np.asarray(noise)).all())


if __name__ == "__main__":
    unittest.main()
