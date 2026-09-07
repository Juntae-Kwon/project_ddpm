"""Run a tiny end-to-end conditional DDPM check on the current JAX device."""

from datetime import datetime
import json
import os
from pathlib import Path

from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from ddpm.data import load_mnist, one_hot
from ddpm.diffusion import make_linear_schedule
from ddpm.model import UNet
from ddpm.training import run_training
from scripts.sample_conditional import save_conditional_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / f"conditional_smoke_{timestamp}_{os.getpid()}"

    all_images, all_labels = load_mnist()
    indices = np.array([np.flatnonzero(all_labels == digit)[0] for digit in range(10)])
    images = all_images[indices]
    labels = all_labels[indices]
    covariates = one_hot(labels)
    assert images.shape == (10, 28, 28, 1)
    assert labels.shape == (10,)
    assert np.array_equal(labels, np.arange(10))
    assert covariates.shape == (10, 10)
    assert covariates.dtype == np.float32
    assert np.array_equal(covariates, np.eye(10, dtype=np.float32))

    model = UNet(base_channels=8)
    schedule = make_linear_schedule(4)
    settings = {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "checkpoint_every": 1,
        "log_every": 1,
        "sample_every": 2,
        "num_samples": 10,
        "seed": 0,
        "labels": labels,
    }
    first = run_training(
        model, schedule, images, run_dir, max_steps=2, **settings
    )
    assert int(first.state.step) == 2
    resumed = run_training(
        model, schedule, images, run_dir, max_steps=3, **settings
    )
    assert int(resumed.state.step) == 3

    parameter_paths = flatten_dict(resumed.state.params)
    assert any("covariate_projection" in path for path in parameter_paths)
    probe_images = jnp.repeat(
        jnp.asarray(images[:1], dtype=jnp.float32) / 127.5 - 1.0, 2, axis=0
    )
    probe_timesteps = jnp.zeros((2,), dtype=jnp.int32)
    probe_covariates = jax.device_put(one_hot(np.array([0, 1])))
    predictions = model.apply(
        {"params": resumed.state.params},
        probe_images,
        probe_timesteps,
        probe_covariates,
    )
    assert predictions.shape == probe_images.shape
    assert np.isfinite(np.asarray(predictions)).all()
    assert not np.allclose(predictions[0], predictions[1])

    generated_path = run_dir / "samples/samples_00000002.npy"
    generated = np.load(generated_path)
    assert generated.shape == (10, 28, 28, 1)
    assert np.isfinite(generated).all()
    grid_dir = run_dir / "conditional_grid"
    grid_path = save_conditional_samples(
        generated, np.arange(10), grid_dir, samples_per_label=1
    )
    assert grid_path.is_file()
    assert len(list(grid_dir.glob("label_*_sample_*.png"))) == 10

    records = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    losses = np.array(
        [record["loss"] for record in records if record["event"] == "train"]
    )
    assert losses.shape == (3,)
    assert np.isfinite(losses).all()
    assert np.all((losses > 0) & (losses < 100))
    assert any(record["event"] == "resume" for record in records)
    assert (run_dir / "checkpoints/checkpoint_00000003.msgpack").is_file()

    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"Labels: {labels.tolist()}", flush=True)
    print(f"One-hot shape: {covariates.shape}", flush=True)
    print(f"Prediction shape: {predictions.shape}", flush=True)
    print(f"Smoke losses: {losses.tolist()}", flush=True)
    print(f"Conditional grid: {grid_path}", flush=True)
    print(f"Smoke-test artifacts: {run_dir}", flush=True)
    print("Conditional smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
