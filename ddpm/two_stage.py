"""Data partition and fixed mathematical settings for the two-stage experiment."""

import hashlib
from pathlib import Path

import numpy as np

from ddpm.diffusion import make_linear_schedule
from scripts.train import ensure_run_config


def experiment_schedule():
    """100 transitions; array index t-1 represents mathematical time t."""
    return make_linear_schedule(100, beta_start=0.001, beta_end=0.2)


def split_indices(seed, pilot_seed, n_stage2):
    if not 1 <= n_stage2 <= 30000:
        raise ValueError("n_stage2 must be between 1 and 30000")
    order = np.random.default_rng(seed).permutation(60000)
    first, second = order[:30000], order[30000:]
    pilot = np.random.default_rng(pilot_seed).permutation(second)[:n_stage2]
    return first, second, pilot


def prepare_experiment(directory, split_seed=20261002, pilot_seed=64, n_stage2=64):
    directory = Path(directory)
    config = dict(split_seed=split_seed, pilot_seed=pilot_seed, n_stage2=n_stage2,
                  T=100, beta_start=0.001, beta_end=0.2,
                  reverse_variance="beta", terminal_covariance="identity",
                  normalization="uint8 / 127.5 - 1", labels_used=False)
    ensure_run_config(directory / "experiment.json", config)
    arrays = dict(zip(("stage1_indices", "stage2_indices", "pilot_indices"),
                      split_indices(split_seed, pilot_seed, n_stage2)))
    for name, values in arrays.items():
        path = directory / f"{name}.npy"
        if path.exists():
            np.testing.assert_array_equal(np.load(path), values)
        else:
            np.save(path, values)
    schedule = experiment_schedule()
    if not (directory / "schedule.npz").exists():
        np.savez(directory / "schedule.npz", betas=np.asarray(schedule.betas),
                 alphas=np.asarray(schedule.alphas), alpha_bars=np.asarray(schedule.alpha_bars))
    terminal = float(schedule.alpha_bars[-1])
    ensure_run_config(directory / "schedule_summary.json", {"alpha_bar_T": terminal})
    print(f"T=100; variance=beta; alpha_bar_T={terminal:.10g}", flush=True)
    return arrays


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
