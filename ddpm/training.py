"""The DDPM noise-prediction objective and one optimizer update."""

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from ddpm.diffusion import sample_forward


def create_train_state(model, key, example_images, learning_rate):
    """Initialize model parameters and an Adam optimizer on JAX's device."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    example_timesteps = jnp.zeros((example_images.shape[0],), dtype=jnp.int32)
    parameters = model.init(key, example_images, example_timesteps)["params"]
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=parameters,
        tx=optax.adam(learning_rate),
    )


def noise_prediction_loss(params, apply_fn, schedule, clean_images, key):
    """Compute E[||epsilon - epsilon_theta(x_t, t)||^2]."""
    timestep_key, noise_key = jax.random.split(key)
    timesteps = jax.random.randint(
        timestep_key,
        (clean_images.shape[0],),
        minval=0,
        maxval=len(schedule.betas),
        dtype=jnp.int32,
    )
    noisy_images, target_noise = sample_forward(
        schedule, clean_images, timesteps, noise_key
    )
    predicted_noise = apply_fn({"params": params}, noisy_images, timesteps)
    return jnp.mean((predicted_noise - target_noise) ** 2)


def train_step(state, schedule, clean_images, key):
    """Take one gradient step and return the updated state and pre-update loss."""

    def loss_fn(params):
        return noise_prediction_loss(
            params, state.apply_fn, schedule, clean_images, key
        )

    loss, gradients = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=gradients), loss
