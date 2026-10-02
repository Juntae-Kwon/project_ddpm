"""The DDPM objective, optimizer update, and resumable training loop."""

import json
import os
from pathlib import Path

from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ddpm.checkpoint import TrainingSnapshot, load_latest_checkpoint, save_checkpoint
from ddpm.data import batches, conditional_batches, one_hot
from ddpm.diffusion import sample_forward
from ddpm.sampling import sample


def create_train_state(
    model, key, example_images, learning_rate, covariates=None
):
    """Initialize model parameters and an Adam optimizer on JAX's device."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    example_timesteps = jnp.zeros((example_images.shape[0],), dtype=jnp.int32)
    if covariates is None:
        parameters = model.init(key, example_images, example_timesteps)["params"]
    else:
        parameters = model.init(
            key, example_images, example_timesteps, covariates
        )["params"]
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=parameters,
        tx=optax.adam(learning_rate),
    )


def noise_prediction_loss(
    params, apply_fn, schedule, clean_images, key, covariates=None
):
    """Compute the DDPM noise-prediction MSE, optionally conditioned on labels."""
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
    if covariates is None:
        predicted_noise = apply_fn({"params": params}, noisy_images, timesteps)
    else:
        predicted_noise = apply_fn(
            {"params": params}, noisy_images, timesteps, covariates
        )
    return jnp.mean((predicted_noise - target_noise) ** 2)


def train_step(state, schedule, clean_images, key, covariates=None):
    """Take one gradient step and return the updated state and pre-update loss."""

    def loss_fn(params):
        return noise_prediction_loss(
            params, state.apply_fn, schedule, clean_images, key, covariates
        )

    loss, gradients = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=gradients), loss


def append_log(path, record):
    """Persist one JSON Lines record and immediately mirror it to stdout."""
    line = json.dumps(record, sort_keys=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(line + "\n")
        log_file.flush()
        os.fsync(log_file.fileno())
    print(line, flush=True)


def _save_samples(path, images):
    """Atomically persist generated samples as a NumPy array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    try:
        with temporary.open("wb") as sample_file:
            np.save(sample_file, np.asarray(images))
            sample_file.flush()
            os.fsync(sample_file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(
    model,
    schedule,
    images,
    run_dir,
    *,
    batch_size,
    learning_rate,
    max_steps,
    checkpoint_every,
    log_every,
    sample_every,
    num_samples,
    seed=0,
    labels=None,
    reverse_variance="posterior",
):
    """Train until max_steps, resuming the newest valid checkpoint if present."""
    positive_settings = {
        "batch_size": batch_size,
        "max_steps": max_steps,
        "checkpoint_every": checkpoint_every,
        "log_every": log_every,
        "sample_every": sample_every,
        "num_samples": num_samples,
    }
    for name, value in positive_settings.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if len(images) < 1:
        raise ValueError("images must not be empty")
    if labels is not None and len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "train.jsonl"
    if labels is None:
        example_images = next(
            batches(images, min(batch_size, len(images)), seed=seed)
        )
        example_covariates = None
    else:
        example_images, example_covariates = next(
            conditional_batches(
                images, labels, min(batch_size, len(images)), seed=seed
            )
        )

    key = jax.random.key(seed)
    key, initialization_key = jax.random.split(key)
    state = create_train_state(
        model,
        initialization_key,
        example_images,
        learning_rate,
        example_covariates,
    )
    snapshot = TrainingSnapshot(
        state=state,
        rng_key_data=jax.random.key_data(key),
        epoch=0,
        batch_in_epoch=0,
    )
    snapshot, restored_path = load_latest_checkpoint(snapshot, checkpoint_dir)
    key = jax.random.wrap_key_data(snapshot.rng_key_data)

    append_log(
        log_path,
        {
            "event": "resume" if restored_path else "start",
            "step": int(snapshot.state.step),
            "checkpoint": str(restored_path) if restored_path else None,
            "max_steps": max_steps,
        },
    )
    compiled_train_step = jax.jit(train_step)

    while int(snapshot.state.step) < max_steps:
        epoch = snapshot.epoch
        completed_batches = snapshot.batch_in_epoch
        epoch_batches = (
            batches(images, batch_size, seed=seed + epoch)
            if labels is None
            else conditional_batches(
                images, labels, batch_size, seed=seed + epoch
            )
        )
        for batch_index, batch in enumerate(epoch_batches):
            if batch_index < completed_batches:
                continue

            if labels is None:
                clean_images, covariates = batch, None
            else:
                clean_images, covariates = batch

            key, step_key = jax.random.split(key)
            state, loss = compiled_train_step(
                snapshot.state, schedule, clean_images, step_key, covariates
            )
            snapshot = TrainingSnapshot(
                state=state,
                rng_key_data=jax.random.key_data(key),
                epoch=epoch,
                batch_in_epoch=batch_index + 1,
            )
            step = int(state.step)

            if step % log_every == 0:
                append_log(
                    log_path,
                    {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_index + 1,
                        "loss": float(loss),
                    },
                )
            if step % checkpoint_every == 0:
                save_checkpoint(snapshot, checkpoint_dir)
            if step % sample_every == 0:
                sample_key = jax.random.fold_in(key, step)
                sample_covariates = None
                if labels is not None:
                    sample_labels = np.arange(num_samples) % 10
                    sample_covariates = jax.device_put(one_hot(sample_labels))
                generated = sample(
                    state.apply_fn,
                    state.params,
                    schedule,
                    sample_key,
                    num_samples=num_samples,
                    image_shape=tuple(clean_images.shape[1:]),
                    covariates=sample_covariates,
                    variance=reverse_variance,
                )
                sample_path = run_dir / "samples" / f"samples_{step:08d}.npy"
                _save_samples(sample_path, generated)
                append_log(
                    log_path,
                    {"event": "sample", "step": step, "path": str(sample_path)},
                )

            if step >= max_steps:
                save_checkpoint(snapshot, checkpoint_dir)
                append_log(log_path, {"event": "complete", "step": step})
                return snapshot

        snapshot = snapshot.replace(epoch=epoch + 1, batch_in_epoch=0)

    return snapshot
