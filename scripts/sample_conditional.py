"""Generate label-conditioned MNIST samples from the latest checkpoint."""

import argparse
import json
from pathlib import Path
import struct
import zlib

import jax
import jax.numpy as jnp
import numpy as np

from ddpm.checkpoint import TrainingSnapshot, load_latest_checkpoint
from ddpm.data import one_hot
from ddpm.diffusion import make_linear_schedule
from ddpm.model import UNet
from ddpm.sampling import sample
from ddpm.training import create_train_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-label", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _write_grayscale_png(path, image):
    """Write one uint8 grayscale image without an image-library dependency."""
    height, width = image.shape

    def chunk(kind, data):
        payload = kind + data
        checksum = struct.pack(">I", zlib.crc32(payload))
        return struct.pack(">I", len(data)) + payload + checksum

    rows = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def save_conditional_samples(images, labels, output_dir, samples_per_label):
    """Save raw samples, individual PNGs, and a label-row grid."""
    images = np.asarray(images)
    labels = np.asarray(labels)
    expected_count = 10 * samples_per_label
    if images.shape != (expected_count, 28, 28, 1):
        raise ValueError(f"unexpected sample shape: {images.shape}")
    expected_labels = np.repeat(np.arange(10), samples_per_label)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("labels must be ordered in rows from 0 through 9")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "samples.npy", images)
    np.save(output_dir / "labels.npy", labels)

    pixels = np.clip((images[..., 0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    for index, (label, image) in enumerate(zip(labels, pixels)):
        column = index % samples_per_label
        _write_grayscale_png(
            output_dir / f"label_{int(label)}_sample_{column:02d}.png", image
        )

    grid = (
        pixels.reshape(10, samples_per_label, 28, 28)
        .transpose(0, 2, 1, 3)
        .reshape(280, 28 * samples_per_label)
    )
    grid_path = output_dir / f"grid_10x{samples_per_label}.png"
    _write_grayscale_png(grid_path, grid)
    return grid_path


def main():
    args = parse_args()
    if args.samples_per_label < 1:
        raise ValueError("samples-per-label must be positive")

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("conditional") is not True:
        raise ValueError("run directory is not for a conditional model")

    model = UNet(base_channels=config["base_channels"])
    schedule = make_linear_schedule(
        config["diffusion_steps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    )
    initialization_key = jax.random.key(config["seed"])
    state = create_train_state(
        model,
        initialization_key,
        jnp.zeros((1, 28, 28, 1), dtype=jnp.float32),
        config["learning_rate"],
        jax.device_put(one_hot(np.array([0], dtype=np.uint8))),
    )
    template = TrainingSnapshot(
        state=state,
        rng_key_data=jax.random.key_data(initialization_key),
        epoch=0,
        batch_in_epoch=0,
    )
    snapshot, checkpoint_path = load_latest_checkpoint(
        template, args.run_dir / "checkpoints"
    )
    if checkpoint_path is None:
        raise FileNotFoundError(f"no valid checkpoint found in {args.run_dir}")

    labels = np.repeat(np.arange(10, dtype=np.uint8), args.samples_per_label)
    covariates = jax.device_put(one_hot(labels))
    generated = sample(
        model.apply,
        snapshot.state.params,
        schedule,
        jax.random.key(args.seed),
        num_samples=len(labels),
        covariates=covariates,
    )
    grid_path = save_conditional_samples(
        generated, labels, args.output_dir, args.samples_per_label
    )
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"samples: {args.output_dir}", flush=True)
    print(f"grid: {grid_path}", flush=True)


if __name__ == "__main__":
    main()
