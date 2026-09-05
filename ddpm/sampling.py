"""Ancestral sampling from the learned DDPM reverse process."""

from functools import partial

import jax
import jax.numpy as jnp


def reverse_mean(schedule, noisy_images, predicted_noise, timestep):
    """Compute the mean of p_theta(x_{t-1} | x_t)."""
    beta = schedule.betas[timestep]
    alpha = schedule.alphas[timestep]
    alpha_bar = schedule.alpha_bars[timestep]
    return (
        noisy_images - beta * predicted_noise / jnp.sqrt(1.0 - alpha_bar)
    ) / jnp.sqrt(alpha)


def p_sample_step(apply_fn, params, schedule, noisy_images, timestep, key):
    """Draw one reverse-process step, omitting random noise when t is zero."""
    timesteps = jnp.full(
        (noisy_images.shape[0],), timestep, dtype=jnp.int32
    )
    predicted_noise = apply_fn({"params": params}, noisy_images, timesteps)
    mean = reverse_mean(schedule, noisy_images, predicted_noise, timestep)

    alpha_bar = schedule.alpha_bars[timestep]
    alpha_bar_previous = jnp.where(
        timestep > 0, schedule.alpha_bars[timestep - 1], 1.0
    )
    posterior_variance = (
        schedule.betas[timestep]
        * (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar)
    )
    noise = jax.random.normal(key, noisy_images.shape, dtype=noisy_images.dtype)
    return mean + jnp.where(
        timestep > 0, jnp.sqrt(posterior_variance) * noise, 0.0
    )


@partial(jax.jit, static_argnames=("apply_fn", "num_samples", "image_shape"))
def sample(apply_fn, params, schedule, key, num_samples, image_shape=(28, 28, 1)):
    """Start from Gaussian noise and run every reverse diffusion step."""
    initial_key, loop_key = jax.random.split(key)
    images = jax.random.normal(
        initial_key, (num_samples,) + image_shape, dtype=jnp.float32
    )
    num_steps = len(schedule.betas)

    def reverse_step(index, carry):
        current_images, current_key = carry
        current_key, step_key = jax.random.split(current_key)
        timestep = num_steps - 1 - index
        current_images = p_sample_step(
            apply_fn,
            params,
            schedule,
            current_images,
            timestep,
            step_key,
        )
        return current_images, current_key

    images, _ = jax.lax.fori_loop(
        0, num_steps, reverse_step, (images, loop_key)
    )
    return images
