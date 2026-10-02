"""Stage 1: fresh unconditional training on the first MNIST half only."""

import argparse
import jax
import numpy as np
from pathlib import Path

from ddpm.data import load_mnist_images
from ddpm.model import UNet
from ddpm.training import run_training
from ddpm.sampling import sample
from ddpm.two_stage import prepare_experiment, experiment_schedule, sha256
from scripts.train import ensure_run_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--split-seed", type=int, default=20261002)
    parser.add_argument("--pilot-seed", type=int, default=64)
    parser.add_argument("--n-stage2", type=int, default=64)
    args = parser.parse_args()
    indices = prepare_experiment(args.experiment_dir, args.split_seed,
                                 args.pilot_seed, args.n_stage2)
    images = load_mnist_images()[indices["stage1_indices"]]
    assert images.shape == (30000, 28, 28, 1)
    run_dir = args.experiment_dir / "stage1"
    config = dict(base_channels=64, batch_size=128, learning_rate=0.0002,
                  seed=0, T=100, beta_start=0.001, beta_end=0.2,
                  reverse_variance="beta", labels_used=False,
                  split_sha256=sha256(args.experiment_dir / "stage1_indices.npy"))
    ensure_run_config(run_dir / "config.json", config)
    model = UNet(base_channels=64)
    schedule = experiment_schedule()
    snapshot = run_training(
        model, schedule, images, run_dir, batch_size=128, learning_rate=0.0002,
        max_steps=args.max_steps, checkpoint_every=2000, log_every=100,
        sample_every=10000, num_samples=16, seed=0, reverse_variance="beta",
    )
    samples = np.asarray(sample(model.apply, snapshot.state.params, schedule,
                                jax.random.key(17), 16, variance="beta"))
    assert samples.shape == (16, 28, 28, 1) and np.isfinite(samples).all()
    np.save(run_dir / "final_ancestral_smoke.npy", samples)
    if int(snapshot.state.step) != args.max_steps:
        raise ValueError("checkpoint exceeds requested training target")
    checkpoint = run_dir / "checkpoints" / f"checkpoint_{args.max_steps:08d}.msgpack"
    ensure_run_config(run_dir / "completed.json", {
        "step": args.max_steps, "checkpoint": str(checkpoint.resolve()),
        "sha256": sha256(checkpoint), "ancestral_smoke_seed": 17,
        "config": config,
    })
    print("Stage 1 complete; parameters frozen for Stage 2.", flush=True)


if __name__ == "__main__":
    main()
