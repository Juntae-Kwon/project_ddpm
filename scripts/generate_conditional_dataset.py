"""Generate a balanced classifier dataset from a trained conditional DDPM."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ddpm.checkpoint import TrainingSnapshot, load_latest_checkpoint
from ddpm.data import one_hot
from ddpm.diffusion import make_linear_schedule
from ddpm.model import UNet
from ddpm.sampling import sample
from ddpm.training import create_train_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-label", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_label < 1 or args.batch_size < 1:
        raise ValueError("sample counts and batch size must be positive")
    config = json.loads((args.run_dir / "config.json").read_text())
    if config.get("conditional") is not True:
        raise ValueError("a conditional run is required")
    model = UNet(base_channels=config["base_channels"])
    key = jax.random.key(args.seed)
    state = create_train_state(
        model, key, jnp.zeros((1, 28, 28, 1)), config["learning_rate"],
        jax.device_put(one_hot(np.array([0]))),
    )
    template = TrainingSnapshot(state, jax.random.key_data(key), 0, 0)
    snapshot, checkpoint = load_latest_checkpoint(template, args.run_dir / "checkpoints")
    if checkpoint is None:
        raise FileNotFoundError("no valid conditional checkpoint")
    schedule = make_linear_schedule(
        config["diffusion_steps"], beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    )
    output = args.output_root / f"evaluation_step_{int(snapshot.state.step):08d}"
    # Refuse to overwrite an existing dataset, including an interrupted run.
    output.mkdir(parents=True, exist_ok=False)
    labels = np.repeat(np.arange(10, dtype=np.uint8), args.samples_per_label)
    images = np.empty((len(labels), 28, 28, 1), dtype=np.float32)
    for start in range(0, len(labels), args.batch_size):
        batch_labels = labels[start:start + args.batch_size]
        key, sample_key = jax.random.split(key)
        generated = sample(
            model.apply, snapshot.state.params, schedule, sample_key,
            num_samples=len(batch_labels),
            covariates=jax.device_put(one_hot(batch_labels)),
        )
        batch = np.asarray(generated)
        if not np.isfinite(batch).all():
            raise ValueError(f"nonfinite generated images at offset {start}")
        images[start:start + len(batch)] = batch
        print(f"Generated {start + len(batch)}/{len(labels)}", flush=True)
    np.save(output / "images.npy", images)
    np.save(output / "labels.npy", labels)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "step": int(snapshot.state.step), "config": config,
        "seed": args.seed, "batch_size": args.batch_size,
        "samples_per_label": args.samples_per_label,
        "shape": list(images.shape), "dtype": str(images.dtype),
        "image_scale": "raw DDPM output on training scale [-1,1]; may exceed bounds",
        "labels": "requested conditioning digits; not independently annotated",
    }
    # This completion marker is written only after both arrays are saved.
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Dataset complete: {output}", flush=True)


if __name__ == "__main__":
    main()
