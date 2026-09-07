"""Command-line entry point for resumable MNIST DDPM training."""

import argparse
import json
import os
from pathlib import Path

from ddpm.data import load_mnist
from ddpm.diffusion import make_linear_schedule
from ddpm.model import UNet
from ddpm.training import run_training


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--base-channels", type=int, required=True)
    parser.add_argument("--diffusion-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument("--log-every", type=int, required=True)
    parser.add_argument("--sample-every", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="condition the denoiser on MNIST digit labels",
    )
    return parser.parse_args()


def ensure_run_config(path, config):
    """Prevent a resumed optimizer state from using incompatible settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Run configuration does not match {path}")
        return

    temporary = path.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)
            config_file.write("\n")
            config_file.flush()
            os.fsync(config_file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    fixed_config = {
        "batch_size": args.batch_size,
        "base_channels": args.base_channels,
        "diffusion_steps": args.diffusion_steps,
        "learning_rate": args.learning_rate,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "seed": args.seed,
    }
    if args.conditional:
        fixed_config["conditional"] = True
    ensure_run_config(args.run_dir / "config.json", fixed_config)

    images, labels = load_mnist()
    model = UNet(base_channels=args.base_channels)
    schedule = make_linear_schedule(
        args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    run_training(
        model,
        schedule,
        images,
        args.run_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        sample_every=args.sample_every,
        num_samples=args.num_samples,
        seed=args.seed,
        labels=labels if args.conditional else None,
    )


if __name__ == "__main__":
    main()
