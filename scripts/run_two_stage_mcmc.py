"""Run or resume the frozen-DDPM MALA-within-Gibbs experiment on a GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from ddpm.checkpoint import TrainingSnapshot
from ddpm.data import load_mnist_images
from ddpm.mcmc import initialize_paths, make_sweep, adapt_epsilons
from ddpm.model import UNet
from ddpm.training import create_train_state, append_log
from ddpm.two_stage import experiment_schedule, sha256
from scripts.train import ensure_run_config


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_chain(directory, state, config):
    path = directory / f"chain_{int(state['sweep']):06d}.npz"
    atomic_npz(path, **state, config_json=np.array(json.dumps(config, sort_keys=True)))
    # Keep two recoverable generations, rather than every full latent path.
    for old in sorted(directory.glob("chain_*.npz"))[:-2]:
        old.unlink()
    return path


def load_chain(directory, config):
    for path in sorted(directory.glob("chain_*.npz"), reverse=True):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if json.loads(str(archive['config_json'])) != config:
                    raise ValueError("chain configuration mismatch")
                state = {k: archive[k] for k in archive.files if k != 'config_json'}
            sweep = int(state['sweep'])
            if path.stem != f"chain_{sweep:06d}" or not 0 <= sweep <= config['total_sweeps']:
                raise ValueError("invalid sweep")
            if state['paths'].shape != (100, config['n'], 28, 28, 1):
                raise ValueError("invalid latent shape")
            for name in ['mu', 'paths', 'epsilons']:
                if not np.isfinite(state[name]).all():
                    raise ValueError(f"nonfinite {name}")
            return state, path
        except Exception as error:
            print(f"Skipping invalid chain checkpoint {path}: {error}", flush=True)
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--output-name', default='stage2')
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--total-sweeps', type=int, default=400)
    parser.add_argument('--burn-in', type=int, default=200)
    parser.add_argument('--thinning', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--tau-mu-squared', type=float, default=1.0)
    parser.add_argument('--epsilon-scale', type=float, default=0.1)
    parser.add_argument('--epsilon-file', type=Path)
    parser.add_argument('--no-adapt', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--stop-after', type=int, help='absolute sweep for resume smoke checks')
    parser.add_argument('--max-seconds', type=float, default=19800)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.burn_in < args.total_sweeps and args.thinning > 0
            and args.tau_mu_squared > 0 and args.epsilon_scale > 0 and args.checkpoint_every > 0):
        raise ValueError('invalid MCMC settings')
    experiment = args.experiment_dir
    output = experiment / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / 'checkpoints'
    log_path = output / 'mcmc.jsonl'
    completed = json.loads((experiment / 'stage1/completed.json').read_text())
    training = completed['config']
    if not (training['T'] == 100 and training['beta_start'] == .001
            and training['beta_end'] == .2 and training['reverse_variance'] == 'beta'
            and training['labels_used'] is False and completed['step'] == 80000):
        raise ValueError('Stage 2 requires the completed fresh 80,000-step T=100 model')
    checkpoint = Path(completed['checkpoint'])
    if sha256(checkpoint) != completed['sha256']:
        raise ValueError('Stage-1 checkpoint hash mismatch')
    pilot = np.load(experiment / 'pilot_indices.npy')
    if not 1 <= args.n <= len(pilot):
        raise ValueError('n must fit the saved pilot subset')
    indices = pilot[:args.n]
    first = np.load(experiment / 'stage1_indices.npy')
    second = np.load(experiment / 'stage2_indices.npy')
    assert len(np.intersect1d(first, second)) == 0
    assert np.isin(indices, second).all()
    pixels = load_mnist_images()[indices]
    observed_host = pixels.astype(np.float32) / 127.5 - 1.0
    observed = jax.device_put(observed_host)

    schedule = experiment_schedule()
    model = UNet(base_channels=training['base_channels'])
    key = jax.random.key(args.seed)
    template = TrainingSnapshot(create_train_state(model, key, jnp.zeros((1,28,28,1)),
                                                   training['learning_rate']),
                                jax.random.key_data(key), 0, 0)
    snapshot = serialization.from_bytes(template, checkpoint.read_bytes())
    params = jax.device_put(snapshot.state.params)
    parameter_hash = hashlib.sha256(serialization.to_bytes(params)).hexdigest()
    del snapshot, template
    initial_epsilons = (np.load(args.epsilon_file) if args.epsilon_file
                        else args.epsilon_scale * np.sqrt(np.asarray(schedule.betas)))
    if initial_epsilons.shape != (100,) or not np.isfinite(initial_epsilons).all() or np.any(initial_epsilons <= 0):
        raise ValueError('epsilon vector must contain 100 finite positive values')
    initial_epsilons = initial_epsilons.astype(np.float32)
    config = dict(n=args.n, total_sweeps=args.total_sweeps, burn_in=args.burn_in,
                  thinning=args.thinning, seed=args.seed, tau_mu_squared=args.tau_mu_squared,
                  adapt=not args.no_adapt, target_acceptance=.57,
                  adaptation='log epsilon += 0.5/(sweep+10)^0.6 * (rate-0.57); clip [1e-6,1]',
                  initial_epsilons=initial_epsilons.tolist(), T=100,
                  beta_start=.001, beta_end=.2, alpha_bar_T=float(schedule.alpha_bars[-1]),
                  reverse_variance='beta', Sigma_T='I', initial_mu='zero; first sweep exact Gibbs',
                  stage1_checkpoint=str(checkpoint), checkpoint_sha256=completed['sha256'],
                  parameters_sha256=parameter_hash, observation_indices=indices.tolist(),
                  normalization='uint8 / 127.5 - 1', checkpoint_every=args.checkpoint_every,
                  coordinate_indices=[0, 100, 300, 500, 783], labels_used=False)
    ensure_run_config(output / 'config.json', config)
    np.save(output / 'observation_indices.npy', indices)
    np.save(output / 'observed.npy', observed_host)
    state, restored = load_chain(checkpoint_dir, config)
    if state is None:
        if list(checkpoint_dir.glob('chain_*.npz')):
            raise RuntimeError('No valid chain checkpoint; refusing to reinitialize an existing chain')
        paths, key = initialize_paths(observed, schedule, key)
        state = dict(sweep=np.array(0), mu=np.zeros((28,28,1), np.float32),
                     paths=np.asarray(paths), rng=jax.random.key_data(key), epsilons=initial_epsilons,
                     accepted=np.zeros(100, np.int64), rejected=np.zeros(100, np.int64),
                     retained_accepted=np.zeros(100, np.int64), retained_rejected=np.zeros(100, np.int64),
                     mu_history=np.zeros((args.total_sweeps,28,28,1), np.float32),
                     acceptance_history=np.zeros((args.total_sweeps,100), np.int32),
                     gradient_mean=np.zeros((args.total_sweeps,100), np.float32),
                     gradient_max=np.zeros((args.total_sweeps,100), np.float32),
                     epsilon_history=np.zeros((args.total_sweeps,100), np.float32),
                     finite_history=np.zeros((args.total_sweeps,100), bool),
                     runtime=np.zeros(args.total_sweeps, np.float64))
        save_chain(checkpoint_dir, state, config)
    append_log(log_path, dict(event='resume' if restored else 'start', sweep=int(state['sweep']),
                              checkpoint=str(restored), backend=jax.default_backend(), devices=str(jax.devices())))
    paths = jax.device_put(state['paths'])
    key = jax.random.wrap_key_data(jnp.asarray(state['rng']))
    epsilons = jax.device_put(state['epsilons'])
    sweep_fn = make_sweep(model.apply, schedule)
    started = time.monotonic()
    end = min(args.total_sweeps, args.stop_after or args.total_sweeps)
    for m in range(int(state['sweep']) + 1, end + 1):
        before = time.monotonic()
        mu, paths, key, diagnostics = sweep_fn(params, observed, paths, key, epsilons, args.tau_mu_squared)
        accepted, grad_mean, grad_max, finite = [np.asarray(v) for v in diagnostics]
        mu_host = np.asarray(mu)
        if not (finite.all() and np.isfinite(mu_host).all()):
            append_log(log_path, dict(event='numerical_failure', sweep=m, bad_times=(np.flatnonzero(~finite)+1).tolist()))
            raise FloatingPointError('MALA numerical failure; resume only from the previous valid checkpoint')
        state['mu_history'][m-1] = mu_host
        state['acceptance_history'][m-1] = accepted
        state['gradient_mean'][m-1] = grad_mean
        state['gradient_max'][m-1] = grad_max
        state['epsilon_history'][m-1] = np.asarray(epsilons)
        state['finite_history'][m-1] = finite
        state['accepted'] += accepted
        state['rejected'] += args.n - accepted
        if m > args.burn_in:
            state['retained_accepted'] += accepted
            state['retained_rejected'] += args.n - accepted
        epsilons = adapt_epsilons(epsilons, jnp.asarray(accepted), args.n, m, args.burn_in, not args.no_adapt)
        state['runtime'][m-1] = time.monotonic() - before
        append_log(log_path, dict(event='sweep', sweep=m, seconds=float(state['runtime'][m-1]),
                                  acceptance=float(accepted.mean()/args.n), finite=True,
                                  gradient_max=float(grad_max.max()), mu_norm=float(np.linalg.norm(mu_host))))
        stop_for_time = time.monotonic() - started >= args.max_seconds
        if m % args.checkpoint_every == 0 or m == end or stop_for_time:
            np.testing.assert_array_equal(np.asarray(observed), observed_host)
            assert np.isfinite(np.asarray(paths)).all()
            state.update(sweep=np.array(m), mu=mu_host, paths=np.asarray(paths),
                         rng=np.asarray(jax.random.key_data(key)), epsilons=np.asarray(epsilons))
            saved = save_chain(checkpoint_dir, state, config)
            append_log(log_path, dict(event='checkpoint', sweep=m, path=str(saved)))
        if stop_for_time:
            raise SystemExit(75)
    assert hashlib.sha256(serialization.to_bytes(params)).hexdigest() == parameter_hash
    assert sha256(checkpoint) == completed['sha256']
    count = int(state['sweep'])
    sweeps = np.arange(1, count+1)
    retained = (sweeps > args.burn_in) & ((sweeps - args.burn_in) % args.thinning == 0)
    history = state['mu_history'][:count]
    atomic_npz(output / 'posterior_samples.npz', sweeps=sweeps, mu_all=history,
               retained_sweeps=sweeps[retained], mu_retained=history[retained],
               mu_norm=np.linalg.norm(history.reshape(count,-1),axis=1),
               coordinate_indices=config['coordinate_indices'],
               coordinate_traces=history.reshape(count,-1)[:,config['coordinate_indices']])
    atomic_npz(output / 'diagnostics.npz', acceptance_counts=state['accepted'],
               rejection_counts=state['rejected'], acceptance_rates=state['accepted']/(count*args.n),
               post_burn_in_accepted=state['retained_accepted'], post_burn_in_rejected=state['retained_rejected'],
               final_epsilons=state['epsilons'], acceptance_history=state['acceptance_history'][:count],
               epsilon_history=state['epsilon_history'][:count], gradient_mean=state['gradient_mean'][:count],
               gradient_max=state['gradient_max'][:count], finite=state['finite_history'][:count],
               runtime_seconds=state['runtime'][:count])
    event = 'complete' if count == args.total_sweeps else 'paused'
    total_sweep_seconds = float(state['runtime'][:count].sum())
    append_log(log_path, dict(event=event, sweep=count, retained=int(retained.sum()),
                              total_sweep_seconds=total_sweep_seconds, parameters_unchanged=True))
    if event == 'complete':
        atomic_json(output / 'completed.json', {
            'sweep': count,
            'retained': int(retained.sum()),
            'parameters_unchanged': True,
            'stage1_checkpoint': str(checkpoint),
            'stage1_checkpoint_sha256': completed['sha256'],
            'final_chain_checkpoint': str(
                checkpoint_dir / f"chain_{count:06d}.npz"
            ),
            'posterior_samples': str(output / 'posterior_samples.npz'),
            'diagnostics': str(output / 'diagnostics.npz'),
            'total_sweep_seconds': total_sweep_seconds,
        })


if __name__ == '__main__':
    main()
