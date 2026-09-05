"""The DDPM forward process q(x_t | x_0)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class DiffusionSchedule(NamedTuple):
    """Precomputed coefficients for a discrete forward diffusion process."""

    betas: jax.Array
    alphas: jax.Array
    alpha_bars: jax.Array
    sqrt_alpha_bars: jax.Array
    sqrt_one_minus_alpha_bars: jax.Array


def make_linear_schedule(num_steps, beta_start=1e-4, beta_end=2e-2):
    """Create the linear variance schedule used in the original DDPM paper."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not 0 < beta_start <= beta_end < 1:
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")

    betas = jnp.linspace(beta_start, beta_end, num_steps, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=jnp.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=jnp.sqrt(1.0 - alpha_bars),
    )


def _extract(values, timesteps, image_ndim):
    """Gather one scalar per image and add axes for image broadcasting."""
    return values[timesteps].reshape((timesteps.shape[0],) + (1,) * (image_ndim - 1))


def q_sample(schedule, clean_images, timesteps, noise):
    """Sample x_t directly from x_0 using supplied standard Gaussian noise.

    Timesteps are zero-based: t=0 applies the first, smallest noising step.
    Supplying the noise separately makes the DDPM equation explicit and lets the
    training objective retain the exact noise that the network must predict.
    """
    signal = _extract(schedule.sqrt_alpha_bars, timesteps, clean_images.ndim)
    noise_scale = _extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, clean_images.ndim
    )
    return signal * clean_images + noise_scale * noise


def sample_forward(schedule, clean_images, timesteps, key):
    """Draw Gaussian noise and return both x_t and its training target."""
    noise = jax.random.normal(key, clean_images.shape, dtype=clean_images.dtype)
    return q_sample(schedule, clean_images, timesteps, noise), noise
