"""MALA-within-Gibbs for frozen DDPM latent paths, with fixed observed Y_0.

Paths store only Y_1,...,Y_T in shape (T,n,H,W,C). Mathematical time t is
array index t-1. Every density is summed over pixels, never averaged.
"""

import jax
import jax.numpy as jnp

from ddpm.sampling import reverse_mean


def gaussian_log_kernel(value, mean, variance):
    """Per-observation log Gaussian, omitting state-independent constants."""
    return -0.5 * jnp.sum((value - mean) ** 2, axis=tuple(range(1, value.ndim))) / variance


def predict_mean(apply_fn, params, schedule, images, index):
    timesteps = jnp.full((images.shape[0],), index, dtype=jnp.int32)
    noise = apply_fn({"params": params}, images, timesteps)
    return reverse_mean(schedule, images, noise, index)


def conditional_log_density(y, lower, upper_mean, upper_variance,
                            index, apply_fn, params, schedule):
    """Log p_theta(lower|y) plus the parent factor, one value per observation.

    Terminal parent: N(y; mu_T,I). Interior parent: N(y; mu_theta(upper,t+1),beta[t]).
    upper_mean is constant during this single-site update.
    """
    mean = predict_mean(apply_fn, params, schedule, y, index)
    return (gaussian_log_kernel(lower, mean, schedule.betas[index])
            + gaussian_log_kernel(y, upper_mean, upper_variance))


def gibbs_parameters(terminal, tau_mu_squared):
    variance = 1.0 / (1.0 / tau_mu_squared + terminal.shape[0])
    return variance * terminal.sum(axis=0), variance


def gibbs_mu(terminal, key, tau_mu_squared=1.0):
    mean, variance = gibbs_parameters(terminal, tau_mu_squared)
    return mean + jnp.sqrt(variance) * jax.random.normal(key, mean.shape)


def initialize_paths(observed, schedule, key):
    """One coherent forward Markov trajectory per observation."""
    def transition(carry, coefficients):
        previous, current_key = carry
        current_key, noise_key = jax.random.split(current_key)
        alpha, beta = coefficients
        next_value = (jnp.sqrt(alpha) * previous
                      + jnp.sqrt(beta) * jax.random.normal(noise_key, previous.shape))
        return (next_value, current_key), next_value
    (_, key), paths = jax.lax.scan(transition, (observed, key),
                                   (schedule.alphas, schedule.betas))
    return paths, key


def proposal_log_kernel(destination, source, gradient, epsilon):
    mean = source + (epsilon ** 2 / 2) * gradient
    return gaussian_log_kernel(destination, mean, epsilon ** 2)


def mh_log_ratio(current_log, proposed_log, current, proposed,
                 current_gradient, proposed_gradient, epsilon):
    return (proposed_log - current_log
            + proposal_log_kernel(current, proposed, proposed_gradient, epsilon)
            - proposal_log_kernel(proposed, current, current_gradient, epsilon))


def accept_proposals(current, proposed, log_ratio, uniforms):
    accepted = jnp.isfinite(log_ratio) & (jnp.log(uniforms) < jnp.minimum(0.0, log_ratio))
    mask = accepted.reshape((len(accepted),) + (1,) * (current.ndim - 1))
    return jnp.where(mask, proposed, current), accepted


def mala_update(log_density, current, epsilon, key):
    # This summed gradient gives each independent observation its own score.
    def total(y):
        values = log_density(y)
        return values.sum(), values
    (_, current_log), gradient = jax.value_and_grad(total, has_aux=True)(current)
    noise_key, accept_key = jax.random.split(key)
    proposed = current + epsilon ** 2 / 2 * gradient + epsilon * jax.random.normal(noise_key, current.shape)
    (_, proposed_log), proposed_gradient = jax.value_and_grad(total, has_aux=True)(proposed)
    log_ratio = mh_log_ratio(current_log, proposed_log, current, proposed,
                             gradient, proposed_gradient, epsilon)
    uniforms = jax.random.uniform(accept_key, (current.shape[0],))
    updated, accepted = accept_proposals(current, proposed, log_ratio, uniforms)
    norms = jnp.linalg.norm(gradient.reshape(current.shape[0], -1), axis=1)
    proposed_norms = jnp.linalg.norm(proposed_gradient.reshape(current.shape[0], -1), axis=1)
    finite = (jnp.all(jnp.isfinite(current_log)) & jnp.all(jnp.isfinite(proposed_log))
              & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(proposed_gradient))
              & jnp.all(jnp.isfinite(proposed)) & jnp.all(jnp.isfinite(log_ratio)))
    return updated, accepted, norms.mean(), jnp.maximum(norms.max(), proposed_norms.max()), finite


def make_sweep(apply_fn, schedule):
    """Compile one systematic sweep. Theta is an input constant, never optimized."""
    @jax.jit
    def sweep(params, observed, paths, key, epsilons, tau_mu_squared):
        params = jax.tree.map(jax.lax.stop_gradient, params)
        key, mu_key = jax.random.split(key)
        mu = gibbs_mu(paths[-1], mu_key, tau_mu_squared)
        T = len(schedule.betas)

        def step(carry, index):
            values, current_key = carry
            current_key, update_key = jax.random.split(current_key)
            lower = jax.lax.cond(index == 0, lambda: observed, lambda: values[index - 1])
            upper_mean, upper_var = jax.lax.cond(
                index == T - 1,
                lambda: (jnp.broadcast_to(mu, observed.shape), jnp.asarray(1.0)),
                lambda: (predict_mean(apply_fn, params, schedule, values[index + 1], index + 1),
                         schedule.betas[index + 1]),
            )
            def log_density(y):
                return conditional_log_density(y, lower, upper_mean, upper_var,
                                               index, apply_fn, params, schedule)
            updated, accepted, norm_mean, norm_max, finite = mala_update(
                log_density, values[index], epsilons[index], update_key)
            values = values.at[index].set(updated)
            return (values, current_key), (accepted.sum(), norm_mean, norm_max, finite)

        (paths, key), diagnostics = jax.lax.scan(step, (paths, key), jnp.arange(T - 1, -1, -1))
        # Store diagnostics in ascending mathematical time 1,...,T.
        return mu, paths, key, tuple(x[::-1] for x in diagnostics)
    return sweep


def adapt_epsilons(epsilons, accepted, n, sweep_number, burn_in, enabled=True):
    """Diminishing log-scale adaptation; never called past burn-in."""
    if not enabled or sweep_number > burn_in:
        return epsilons
    gain = 0.5 / (sweep_number + 10) ** 0.6
    return jnp.clip(epsilons * jnp.exp(gain * (accepted / n - 0.57)), 1e-6, 1.0)
