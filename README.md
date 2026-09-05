# DDPM on MNIST with JAX

A minimal learning project for Denoising Diffusion Probabilistic Models.
JAX handles array computation and differentiation, Flax defines the network,
and Optax provides optimization. NumPy will handle host-side data preparation.

## Current milestone

Project and environment setup only. The DDPM data pipeline, diffusion process,
network, training, sampling, and resumable checkpoints will be added in separate
verified milestones. No MNIST training has been run.

## Environment

Run from Blue storage. Python 3.11 is available through HiPerGator's module system.
For a fresh environment:

```bash
cd /blue/ark007/juntaekwon/projects/DDPM
module load python/3.11
# The module sets PYTHONHOME; a virtual environment must use its own prefix.
unset PYTHONHOME
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For subsequent sessions, use:

```bash
cd /blue/ark007/juntaekwon/projects/DDPM
unset PYTHONHOME
source .venv/bin/activate
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export PYTHONUNBUFFERED=1
```

Keep datasets in `data/` and generated checkpoints, logs, and samples in `runs/`.
These directories, `.venv/`, and `.cache/` are excluded from Git, as is `AGENTS.md`.

## Verify setup

```bash
python -m pip check
python -u scripts/check_environment.py
```

The check prints package versions and available JAX devices, initializes a tiny
Flax dense layer, runs a JIT-compiled loss and gradient computation, and applies
one Optax update. It checks shapes, finite values, and device placement. This is
an environment check; the later DDPM smoke test will exercise the full pipeline.

The initial installation uses ordinary JAX packages without CUDA extras. CPU
execution in the tunnel is expected. The code does not force a CPU backend;
JAX selects the available default device. A later GPU Slurm job must install or
verify appropriate CUDA-enabled JAX support and explicitly confirm GPU use before
training. Merely allocating a GPU is not sufficient for this initial installation.

## Remaining milestones

1. MNIST loading and normalization to `[-1, 1]`.
2. Forward diffusion and its noise schedule.
3. A small timestep-conditioned Flax U-Net predicting noise.
4. Noise-prediction MSE and optimizer updates.
5. DDPM reverse sampling.
6. Training with periodic resumable checkpoints and persisted, flushed logs.
7. A tiny end-to-end smoke test and documentation for the later GPU workflow.

Final training settings will be selected after inspecting the actual Slurm GPU,
memory, CPU, RAM, and wall-time allocation. No full-training job is created or
submitted at this stage.
