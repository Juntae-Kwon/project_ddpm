# DDPM on MNIST with JAX

A minimal learning project for Denoising Diffusion Probabilistic Models.
JAX handles array computation and differentiation, Flax defines the network,
and Optax provides optimization. NumPy will handle host-side data preparation.

## Current milestone

Project setup, data, diffusion, model, training, reverse sampling, resumable
checkpoints, and persisted logs are complete. The final small CPU smoke test and
later GPU workflow remain. No full MNIST training has been run.

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

## MNIST data pipeline

`ddpm/data.py` downloads the training images from the Google-hosted MNIST mirror
into this project's `data/` directory and reuses the local file on later calls.
It reads the gzip-compressed IDX format directly using Python and NumPy. Labels
are unnecessary for unconditional generation and are not downloaded.

From the project root with the environment activated:

```python
from ddpm.data import load_mnist, batches

images = load_mnist()  # Host uint8 array: (60000, 28, 28, 1), about 47 MB.
batch = next(batches(images, batch_size=8, seed=0))
print(batch.shape, batch.dtype, batch.devices())
```

Each batch is converted to `float32` and normalized by `pixel / 127.5 - 1`,
mapping black to -1 and white to 1. These are the clean images, called `x_0` in
DDPM. The batch is placed on JAX's default device while the dataset stays in host
memory. The batch size above is only a small data check, not a training setting.

`batches` yields one shuffled epoch. The same seed reproduces the same order;
use a new seed for each epoch. Every image appears once, including a final
partial batch when needed. Its different shape may trigger a separate JIT
compilation in later training code.

Run the tiny offline data tests (fixtures are created under `.cache/tmp/`):

```bash
python -m unittest discover -s tests -v
```

## Forward diffusion

`ddpm/diffusion.py` creates a linear variance schedule and evaluates the DDPM
forward process at any timestep without simulating all earlier steps:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
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

`ddpm/model.py` defines a small Flax U-Net that accepts a noisy image `x_t` and
its timestep `t`, then predicts the Gaussian noise in the image. Fixed sine and
cosine features encode each timestep; a small MLP transforms that encoding and
adds it to every residual block. This lets one network change its prediction
according to the current noise level.

The image path is deliberately compact:

```text
28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28
```

Residual blocks provide the convolutional processing. Skip connections copy
fine spatial information from each downsampling level to the matching upsampling
level. Group normalization has no running statistics, so training and sampling
use the same model state. The final convolution returns one predicted-noise value
for each input pixel.

`base_channels` controls model width and defaults to 32, but the final value will
be selected after inspecting the GPU allocation. It must be a multiple of eight
because each normalization layer uses eight groups. The tests use a width of
eight to keep their CPU work deliberately small.

## Training objective

`ddpm/training.py` implements the simplified DDPM noise-prediction objective:

```text
L = mean((epsilon - epsilon_theta(x_t, t))^2)
```

For every batch, it samples an independent timestep for each image, draws
standard Gaussian noise, constructs `x_t` with the closed-form forward process,
and asks the U-Net to recover that exact noise. Uniform timestep sampling teaches
the same network to denoise throughout the diffusion trajectory.

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
Gaussian noise and visits all timesteps in reverse using `jax.lax.fori_loop`, so
the complete sampling path is compiled and remains on JAX's selected device.
Supplying the same random key reproduces the same samples.

Samples from an untrained network are expected to look like noise. Image quality
will only become meaningful after the later full training run.

## Resumable training

Run training from the project root with every resource-sensitive setting chosen
explicitly. The placeholders below are intentionally not recommendations for the
later full run:

```bash
python -u -m scripts.train \
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

## Remaining milestones

1. A tiny end-to-end smoke test and documentation for the later GPU workflow.

Final training settings will be selected after inspecting the actual Slurm GPU,
memory, CPU, RAM, and wall-time allocation. No full-training job is created or
submitted at this stage.
