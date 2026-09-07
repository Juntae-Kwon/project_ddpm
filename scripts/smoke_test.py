"""Run a deliberately tiny end-to-end DDPM check on JAX's current device."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import jax
import numpy as np

from ddpm.data import load_mnist
from ddpm.diffusion import make_linear_schedule
from ddpm.model import UNet
from ddpm.training import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="new directory under the project (default: unique runs/smoke_*)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or PROJECT_ROOT / "runs" / f"smoke_{timestamp}_{os.getpid()}"
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("smoke-test outputs must remain under the project root")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"smoke-test directory is not empty: {run_dir}")

    images, _ = load_mnist()
    images = images[:8]
    assert images.shape == (8, 28, 28, 1)
    assert images.dtype == np.uint8

    model = UNet(base_channels=8)
    schedule = make_linear_schedule(4)
    settings = {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "checkpoint_every": 1,
        "log_every": 1,
        "sample_every": 2,
        "num_samples": 2,
        "seed": 0,
    }

    first = run_training(
        model, schedule, images, run_dir, max_steps=2, **settings
    )
    assert int(first.state.step) == 2
    resumed = run_training(
        model, schedule, images, run_dir, max_steps=3, **settings
    )
    assert int(resumed.state.step) == 3

    active_devices = resumed.state.step.devices()
    for value in jax.tree.leaves((resumed.state.params, resumed.state.opt_state)):
        if hasattr(value, "dtype"):
            assert np.isfinite(np.asarray(value)).all()
            assert value.devices() == active_devices

    sample_path = run_dir / "samples/samples_00000002.npy"
    generated = np.load(sample_path)
    assert generated.shape == (2, 28, 28, 1)
    assert generated.dtype == np.float32
    assert np.isfinite(generated).all()

    records = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_records = [record for record in records if record["event"] == "train"]
    assert [record["step"] for record in train_records] == [1, 2, 3]
    losses = np.array([record["loss"] for record in train_records])
    assert np.isfinite(losses).all()
    assert np.all(losses > 0)
    assert np.all(losses < 100)
    assert any(record["event"] == "resume" for record in records)
    assert (run_dir / "checkpoints/checkpoint_00000003.msgpack").is_file()
    assert not list(run_dir.rglob("*.tmp"))

    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"Smoke losses: {losses.tolist()}", flush=True)
    print(f"Smoke-test artifacts: {run_dir}", flush=True)
    print("End-to-end smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
