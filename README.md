# DDPM on MNIST with JAX

A minimal learning project for Denoising Diffusion Probabilistic Models.
JAX handles array computation and differentiation, Flax defines the network,
and Optax provides optimization. NumPy will handle host-side data preparation.

## Current milestone

The unconditional model has completed a 40,000-step L4 training run. Label
conditioning is implemented and covered by a small end-to-end smoke test; its
full-run Slurm script is prepared but has not been submitted.

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

The requirements include JAX's pip-managed CUDA 12 libraries, but installation
alone does not prove GPU access. The batch job still requires
`scripts.check_gpu` to pass on its allocated device.

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

CPU execution in the tunnel is expected even after installing the CUDA extra.
The code does not force a CPU backend; JAX selects the available default device.
The Slurm job explicitly confirms CUDA GPU execution before training. Merely
allocating a GPU is not sufficient verification.

## MNIST data pipeline

`ddpm/data.py` downloads the training images and labels from the Google-hosted
MNIST mirror into this project's `data/` directory and reuses the local files on
later calls. It reads the gzip-compressed IDX format directly using Python and
NumPy.

From the project root with the environment activated:

```python
from ddpm.data import conditional_batches, load_mnist

images, labels = load_mnist()
image_batch, covariates = next(
    conditional_batches(images, labels, batch_size=8, seed=0)
)
print(image_batch.shape, covariates.shape)  # (8, 28, 28, 1), (8, 10)
```

Each image batch is converted to `float32` and normalized by
`pixel / 127.5 - 1`, mapping black to -1 and white to 1. Each digit label becomes
a ten-dimensional `float32` one-hot covariate. Both batches are placed on JAX's
default device while the dataset stays in host memory.

`conditional_batches` applies the same permutation to images and labels, so they
remain aligned during each shuffled epoch. `batches` remains available for the
unconditional path.

Run the tiny offline data tests (fixtures are created under `.cache/tmp/`):

```bash
python -m unittest discover -s tests -v
```

## Forward diffusion

`ddpm/diffusion.py` creates a linear variance schedule and evaluates the DDPM
forward process at any timestep without simulating all earlier steps:

```text
y_t = sqrt(alpha_bar_t) * y_0 + sqrt(1 - alpha_bar_t) * epsilon
epsilon ~ N(0, I)
```

Here `alpha_t = 1 - beta_t` and `alpha_bar_t` is the cumulative product of
`alpha_0` through `alpha_t`. As `alpha_bar_t` decreases, the clean image signal
shrinks and the Gaussian noise contribution grows.

`make_linear_schedule` keeps the number of steps and beta endpoints explicit.
The conventional `1e-4` to `2e-2` defaults are useful for understanding and
testing the formulation; the final timestep count remains a later GPU training
decision. Timesteps are zero-based, so index 0 is the first small noising step.

`sample_forward` returns both `x_t` and the exact sampled noise. That noise will
be the target for the denoising network. All schedule arrays and computations
remain on JAX's selected device and work under `jax.jit`.

## Denoising network

`ddpm/model.py` defines a small Flax U-Net that accepts a noisy image `y_t`, its
timestep `t`, and an optional one-hot covariate `x`. Fixed sine and cosine
features encode the timestep. For conditional calls, one learned linear layer
maps `x` to the same dimension and adds it to the timestep embedding before the
existing residual blocks. The image path itself is unchanged.

The image path is deliberately compact:

```text
28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28
```

Residual blocks provide the convolutional processing. Skip connections copy
fine spatial information from each downsampling level to the matching upsampling
level. Group normalization has no running statistics, so training and sampling
use the same model state. The final convolution returns one predicted-noise value
for each input pixel.

`base_channels` controls model width and defaults to 32. Full training uses 64;
tests use eight to keep their CPU work deliberately small. Unconditional calls
omit `x`, preserving compatibility with the existing unconditional model.

## Training objective

`ddpm/training.py` implements the simplified DDPM noise-prediction objective:

```text
L = mean((epsilon - epsilon_theta(y_t, t, x))^2)
```

For every batch, it samples an independent timestep for each image, draws
standard Gaussian noise, constructs `y_t` with the unchanged closed-form forward
process, and asks the U-Net to recover that exact noise while receiving `x`.
Only the image is diffused; the one-hot covariate remains fixed.

`create_train_state` initializes the model and an Optax Adam optimizer. Its
learning rate is a required argument so the eventual GPU configuration remains
an explicit decision. `train_step` computes gradients and applies one update;
the caller owns the random key and must provide a fresh key for each real
training step. Both functions preserve JAX device placement, and `train_step`
can be compiled with `jax.jit`.

## Reverse sampling

`ddpm/sampling.py` implements the ancestral DDPM reverse process. At timestep
`t`, the U-Net predicts the noise in `x_t`, and the reverse mean is

```text
mu_theta = (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_theta)
           / sqrt(alpha_t)
```

For every step above zero, the sampler adds Gaussian noise scaled by the true
forward posterior variance

```text
beta_tilde_t = beta_t * (1 - alpha_bar_(t-1)) / (1 - alpha_bar_t).
```

No noise is added after the final `t=0` prediction. `sample` begins with standard
Gaussian noise and visits all timesteps in reverse using `jax.lax.fori_loop`.
For conditional generation, the same one-hot covariate is passed at every reverse
step. Supplying the same random key reproduces the same samples.

Samples from an untrained network are expected to look like noise. Image quality
will only become meaningful after the later full training run.

## Resumable training

Run training from the project root with every resource-sensitive setting chosen
explicitly. Pass `--conditional` for label conditioning; omit it to retain the
unconditional path. The placeholders below show the conditional form:

```bash
python -u -m scripts.train \
  --conditional \
  --run-dir runs/NAME \
  --batch-size BATCH_SIZE \
  --base-channels BASE_CHANNELS \
  --diffusion-steps DIFFUSION_STEPS \
  --learning-rate LEARNING_RATE \
  --max-steps MAX_STEPS \
  --checkpoint-every CHECKPOINT_INTERVAL \
  --log-every LOG_INTERVAL \
  --sample-every SAMPLE_INTERVAL \
  --num-samples NUM_SAMPLES
```

Use `python -u` so Slurm captures stdout without Python buffering. Each JSON log
record is also appended to `train.jsonl`, flushed, and synchronized to storage.
Generated sample batches are stored as NumPy arrays for later visualization.

Each checkpoint contains the model parameters, Adam state, completed optimizer
step, next random key, epoch, and completed batch count. Files are written to a
temporary name and atomically renamed. On startup, training scans checkpoints
from newest to oldest; corrupt, partial, or incompatible files produce a warning
and are skipped. Shuffling is deterministic per epoch, so restoration continues
at the next unprocessed batch.

The run directory has this ignored, generated layout:

```text
runs/NAME/
  config.json
  train.jsonl
  checkpoints/checkpoint_STEP.msgpack
  samples/samples_STEP.npy
```

`config.json` protects a resumed optimizer state from changes to model width,
diffusion schedule, learning rate, batch size, or seed. `--max-steps` and output
frequencies may change between invocations, which allows a verified run to be
extended without changing the learned state.

Generate a row for each label from the newest conditional checkpoint with:

```bash
python -u -m scripts.sample_conditional \
  --run-dir runs/mnist_conditional_l4_b128_c64_t1000 \
  --output-dir samples/mnist_conditional \
  --samples-per-label 8 \
  --seed 1
```

The output includes raw arrays, individual PNGs, and `grid_10x8.png` with labels
0 through 9 as rows.

## End-to-end smoke test

Run the deliberately small current-device check from the project root:

```bash
python -u -m scripts.smoke_test
python -u -m scripts.conditional_smoke_test
```

The conditional check selects one real image for every digit, verifies the
one-hot covariates and label-sensitive model output, and runs only three optimizer
updates with four diffusion steps. It also restores a checkpoint, performs
conditional sampling, and writes a ten-row test grid.

Each invocation creates a small unique ignored directory under `runs/smoke_*` so
it never overwrites earlier evidence. Pass `--run-dir` only when a specific new,
empty project-local directory is desired.

## GPU training

The prepared batch script inspects its Slurm allocation, verifies CUDA-enabled
JAX, runs a short GPU smoke test, and resumes the same run into full training.

## Project status

The prepared conditional script is `slurm/train_conditional_ddpm.sbatch`. It
requests one L4 GPU, four CPU cores, 16 GB RAM, and one hour. It performs a
five-step GPU check and resumes the conditional run toward 40,000 total updates.
It has not been submitted. The local `slurm/` directory is excluded from Git.

## Classifier-based conditional evaluation

The notebook `notebooks/conditional_ddpm_evaluation.ipynb` trains a small
Flax CNN on 10,000 generated images and evaluates it on those same images.
Its accuracy is an in-sample fit diagnostic, not independent generative-quality
or held-out accuracy. Classifier inputs are pixels; targets are requested digits.

The ignored `slurm/sample_latest.sbatch` now generates 1,000 images per digit
with `scripts.generate_conditional_dataset`, then runs JupyterLab in the
foreground on the same L4 allocation. The one-hour allocation remains active
until Jupyter is stopped or the wall time expires. The server URL and token
appear in its Slurm error log. Use SSH port forwarding to access the server.

The generator saves `images.npy`, `labels.npy`, and checkpoint metadata under
`samples/conditional/evaluation_step_STEP/`. It refuses to overwrite an existing
dataset. The notebook defaults to step 80000; update its DATA_DIR if using a
different checkpoint. Select the project kernel **DDPM (.venv)** and run cells
in order. Metrics and confusion-matrix artifacts are saved under
`runs/conditional_classifier_evaluation/`. Notebook source is tracked; datasets,
runtime files, and generated evaluation artifacts are ignored.
