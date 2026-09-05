"""Atomic, resumable training checkpoints using Flax serialization."""

import os
from pathlib import Path
import sys

from flax import serialization, struct
import jax
import numpy as np


@struct.dataclass
class TrainingSnapshot:
    """Everything needed to continue at the next unprocessed batch."""

    state: object
    rng_key_data: jax.Array
    epoch: int
    batch_in_epoch: int


def save_checkpoint(snapshot, directory):
    """Atomically save a snapshot named by its completed optimizer step."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    step = int(snapshot.state.step)
    path = directory / f"checkpoint_{step:08d}.msgpack"
    temporary = directory / f".{path.name}.tmp"
    try:
        with temporary.open("wb") as checkpoint_file:
            checkpoint_file.write(serialization.to_bytes(snapshot))
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_latest_checkpoint(snapshot_template, directory):
    """Restore the newest valid compatible checkpoint, or return the template."""
    directory = Path(directory)
    candidates = sorted(directory.glob("checkpoint_*.msgpack"), reverse=True)
    for path in candidates:
        try:
            filename_step = int(path.stem.removeprefix("checkpoint_"))
            snapshot = serialization.from_bytes(snapshot_template, path.read_bytes())
            if int(snapshot.state.step) != filename_step:
                raise ValueError("step does not match checkpoint filename")
            if snapshot.epoch < 0 or snapshot.batch_in_epoch < 0:
                raise ValueError("training position must not be negative")
            key_data = np.asarray(snapshot.rng_key_data)
            if key_data.shape != (2,) or key_data.dtype != np.uint32:
                raise ValueError("invalid random key data")
            return snapshot, path
        except Exception as error:
            # A partial, corrupt, or incompatible checkpoint is not resumable.
            print(f"Skipping invalid checkpoint {path}: {error}", file=sys.stderr, flush=True)
            continue
    return snapshot_template, None
