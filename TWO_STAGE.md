# Two-stage MNIST pilot

This experiment uses only images. It trains a fresh unconditional model on
30,000 training images and freezes it for a latent-variable MCMC pilot on 64
different observations from the other 30,000 images. Outputs are diagnostics
and samples; no posterior interpretation is made.

## Fixed experiment

- Split: `numpy.random.default_rng(20261002).permutation(60000)`; first half is
  Stage 1, second half is Stage 2. Pilot: seed 64 permutes the Stage-2 pool.
- All indices are zero-based positions in the MNIST training image file.
- Both stages normalize uint8 pixels as `pixels / 127.5 - 1`.
- `T=100`, linear beta from 0.001 to 0.2; alpha is `1-beta`, alpha_bar is the
  cumulative product. Saved alpha_bar_T is approximately 0.0000203901.
- Every reverse transition has variance **beta_t**, including `Y_0 | Y_1`.
  The older experiments retain their posterior-variance sampler by default.
- Stage 1: existing width-64 U-Net, Adam 0.0002, batch 128, seed 0, 80,000 total
  updates, checkpoints every 2,000, logs every 100, 16 samples every 10,000.
  The final ordinary sampling check uses seed 17 and includes final-step noise.
- Stage 2: Sigma_T is exactly I, mu_T prior N(0,I), seed 20261003, n=64,
  400 sweeps, burn-in 200, thinning 2. Retain sweeps 202,204,...,400 (100 draws).

## Target and updates

The augmented target is proportional to

```
p(mu_T) product_i N(Y_iT; mu_T, I)
       product_{t=1}^100 N(Y_i,t-1; mu_theta(Y_it,t), beta_t I).
```

The U-Net uses zero-based index `t-1`. `paths[t-1]` holds mathematical Y_t;
observed Y_0 is stored separately and never appears in mutable latent storage.
Initialization runs the actual forward Markov recurrence, not independent
closed-form draws at each time.

Each sweep first draws mu_T from its exact Gaussian full conditional, with
variance `1/(1/tau_mu_squared+n)` and mean equal to that variance times the sum
of terminal states. It then visits t=100,...,1, using the newest neighbors.
For each observation the log conditional contains exactly two factors:

```
t=100: log N(y;mu_T,I) + log N(Y_99;mu_theta(y,100),beta_100 I)
t<100: log N(Y_t-1;mu_theta(y,t),beta_t I)
       + log N(y;mu_theta(Y_t+1,t+1),beta_t+1 I).
```

Gaussian constants can be omitted because their variances and dimensions are
fixed within each MH update. Pixel contributions are **summed**, not averaged.
Gradients differentiate the denoiser with respect to its input. Parameters are
stop-gradient constants and receive no optimizer updates; hashes verify them.

MALA proposes `y' = y + epsilon_t**2/2 * grad_log_pi(y) + epsilon_t*noise`.
Its log acceptance ratio includes the target difference plus
`log q(y|y') - log q(y'|y)`. Decisions use one uniform draw per observation.
The implementation vectorizes observations at each fixed time.

Initial epsilon_t is `0.1*sqrt(beta_t)` (an explicit vector file can override it).
During burn-in only, log epsilon_t is updated after each sweep by
`0.5/(sweep+10)**0.6 * (acceptance_rate_t - 0.57)`, clipped to [1e-6,1].
After sweep 200 the vector is frozen. `--no-adapt` disables adaptation.

## Execution

The ignored batch files are `slurm/two_stage_train.sbatch` and
`slurm/two_stage_mcmc.sbatch`. Both activate `.venv` explicitly, use Blue storage,
verify CUDA, and request hpg-turin, one L4, 4 CPUs, 32 GB, six hours. Stage 2 is
submitted with an `afterok` dependency on Stage 1. No existing T=1000 checkpoint
is used. A completed.json manifest pins the exact Stage-1 checkpoint and hash.

Stage 2 first runs a separate n=4, T=100, three-sweep GPU smoke, stopping after
sweep 1 and restoring for sweeps 2 and 3. Only then does it launch the pilot.

```
python -u -m scripts.train_two_stage \
  --experiment-dir runs/two_stage_mnist_t100 --max-steps 80000
python -u -m scripts.run_two_stage_mcmc \
  --experiment-dir runs/two_stage_mnist_t100 \
  --n 64 --total-sweeps 400 --burn-in 200 --thinning 2
```

For larger n, prepare a larger deterministic pilot subset when creating a new
experiment with `prepare_experiment(..., n_stage2=...)`. Never change a running
chain's subset/configuration. Existing observations remain a deterministic prefix.

## Saved state and diagnostics

Under `runs/two_stage_mnist_t100/`:

- `experiment.json`, `stage1_indices.npy`, `stage2_indices.npy`, `pilot_indices.npy`
- `schedule.npz` and `schedule_summary.json`
- `stage1/config.json`, `train.jsonl`, `checkpoints/`, `completed.json`,
  `final_ancestral_smoke.npy`
- `stage2_smoke/` and `stage2/`, each containing `config.json`, `observed.npy`,
  `observation_indices.npy`, `mcmc.jsonl`, and `checkpoints/chain_*.npz`
- `stage2/posterior_samples.npz`: all mu draws `(400,28,28,1)`, retained draws
  `(100,28,28,1)`, sweep IDs, norm and coordinate traces
- `stage2/diagnostics.npz`: timewise acceptance/rejection totals, post-burn-in
  totals, per-sweep acceptance and epsilon histories, final epsilons,
  gradient norms, finite checks, and sweep timings including compilation

Checkpoints contain current sweep, mu, full latent paths, PRNG, adaptation's
epsilon vector, counters, trace history, and configuration. Atomic writes retain
the newest two generations. Resume restores the same chain, not a fresh path.
The final checkpoint is the final latent state. Invalid checkpoints are reported
and skipped; if all existing checkpoints are invalid, execution stops.

Checkpoints are saved every 10 sweeps. A process stops safely after approximately
5.5 hours if necessary (exit 75); resubmit the same Stage-2 script to continue.
JSONL entries after the last checkpoint may repeat after a restart; checkpoint
trace arrays and final NPZ files provide the authoritative sweep sequence.

Tests: `python -m unittest discover -s tests -v`. The original test suite remains
in place, and `tests/test_mcmc.py` covers the new mathematical invariants and
deterministic checkpoint replay. The GPU smoke additionally tests the actual
trained U-Net input gradients and numerical behavior at T=100.
