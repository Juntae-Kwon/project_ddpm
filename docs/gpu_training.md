# Later GPU training workflow

Use this checklist only inside a GPU-enabled Slurm allocation. It prepares and
verifies the run; it is not a Slurm submission script and does not select final
training settings.

## 1. Enter the project environment

```bash
cd /blue/ark007/juntaekwon/projects/DDPM
module load python/3.11
unset PYTHONHOME
source .venv/bin/activate
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
```

## 2. Inspect the allocation before choosing settings

```bash
hostname
date
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free,driver_version --format=csv
nvidia-smi
nproc
lscpu
free -h
scontrol show job "$SLURM_JOB_ID"
```

Record the GPU model, compute capability, free and total GPU memory, driver,
allocated CPU cores, system RAM, and Slurm time limit in the job log. Confirm
that no unexpected process is consuming the allocation.

## 3. Install and verify the appropriate CUDA-enabled JAX build

Follow the current [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
after inspecting the allocated GPU and driver. JAX currently offers pip-managed
CUDA 12 and CUDA 13 installations and recommends the pip-managed CUDA libraries.
Do not choose between them before seeing the allocation.

For the pip-managed option, select one compatible command and retain this
project's verified JAX version:

```bash
# Choose only after checking the allocated driver and GPU.
python -m pip install --upgrade "jax[cuda13]==0.10.2"
# or: python -m pip install --upgrade "jax[cuda12]==0.10.2"
```

The official guide currently requires an NVIDIA driver of at least 580 for its
CUDA 13 wheel or at least 525 for its CUDA 12 wheel. CUDA 13 also requires GPU
compute capability 7.5 or newer. Recheck the guide at run time because these
requirements can change. For pip-managed CUDA libraries, the guide advises that
`LD_LIBRARY_PATH` should not override them. Follow HiPerGator policy if the site
requires its local CUDA modules instead; JAX documents separate `cuda13-local`
and `cuda12-local` extras for that case.

After installation, verify the environment and require actual GPU execution:

```bash
python -m pip check
python -u scripts/check_environment.py
python -u -m scripts.check_gpu
```

Do not continue if `scripts.check_gpu` reports a CPU backend. It compiles a small
calculation, verifies that the result is resident on a GPU device, and prints the
device kind and available JAX memory statistics.

## 4. Choose a configuration from the allocation

Choose these together after the inspection:

- Batch size and `base_channels` determine most training memory use. Begin with
  a conservative pair, then increase only if the GPU smoke test leaves useful
  headroom.
- Diffusion timesteps affect sampling cost directly and influence how closely
  the implementation follows the gradual DDPM process.
- Choose the Adam learning rate for the selected batch size and model width.
- Convert the useful wall time into a realistic total step count after measuring
  compiled step time. Leave time for final checkpoint and sample writes.
- Set checkpoint frequency so lost work after interruption is acceptable without
  producing excessive files. Set logging and sampling frequencies independently;
  a full reverse sample evaluates the U-Net once per diffusion timestep.

Favor a modest model that converges reliably on MNIST. Additional width is useful
only when the smoke test shows enough memory and step throughput.

## 5. Run a short GPU smoke test with the chosen full-run configuration

Use the intended full-run directory and configuration, but initially set
`--max-steps` to only a few steps. Set `--sample-every` to the final smoke step so
the reverse loop is also exercised on the GPU.

```bash
python -u -m scripts.train \
  --run-dir runs/FULL_RUN_NAME \
  --batch-size CHOSEN_BATCH_SIZE \
  --base-channels CHOSEN_BASE_CHANNELS \
  --diffusion-steps CHOSEN_DIFFUSION_STEPS \
  --learning-rate CHOSEN_LEARNING_RATE \
  --max-steps GPU_SMOKE_STEPS \
  --checkpoint-every GPU_SMOKE_STEPS \
  --log-every 1 \
  --sample-every GPU_SMOKE_STEPS \
  --num-samples CHOSEN_SAMPLE_COUNT
```

Confirm finite losses, GPU device use, peak memory headroom, reasonable step time,
a final checkpoint, persisted JSONL logs, and a finite sample array. If the run
is too large or too slow, use a new run directory with a revised configuration;
the existing `config.json` deliberately rejects incompatible resume settings.

## 6. Resume into full training

Re-run the same command with the same model, data, optimizer, schedule, and seed
settings. Increase `--max-steps` to the resource-aware training target and adjust
only checkpoint, logging, and sampling frequencies as needed. The runner restores
the latest valid checkpoint automatically, including its optimizer, random key,
epoch, and batch position. Verify the `resume` record in `train.jsonl` before
leaving the batch job unattended.

Never infer success from GPU allocation alone: require the CUDA backend check and
the short compiled GPU run before starting full training.
