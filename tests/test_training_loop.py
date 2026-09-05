"""Small integration tests for logging, checkpointing, and resume behavior."""

import json
from pathlib import Path
import tempfile
import unittest

from flax import linen as nn
import jax
import numpy as np

from ddpm.checkpoint import load_latest_checkpoint
from ddpm.diffusion import make_linear_schedule
from ddpm.training import run_training
from scripts.train import ensure_run_config


class TinyNoiseModel(nn.Module):
    """One-parameter predictor that keeps this infrastructure test inexpensive."""

    @nn.compact
    def __call__(self, noisy_images, timesteps):
        del timesteps
        scale = self.param("scale", nn.initializers.zeros_init(), ())
        return scale * noisy_images


class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".cache" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name) / "run"

    def run_until(self, max_steps):
        images = np.arange(6 * 4 * 4, dtype=np.uint8).reshape(6, 4, 4, 1)
        return run_training(
            TinyNoiseModel(),
            make_linear_schedule(2),
            images,
            self.run_dir,
            batch_size=2,
            learning_rate=1e-3,
            max_steps=max_steps,
            checkpoint_every=1,
            log_every=1,
            sample_every=2,
            num_samples=2,
            seed=7,
        )

    def test_training_resumes_latest_valid_checkpoint(self):
        first = self.run_until(2)
        self.assertEqual(int(first.state.step), 2)
        first_checkpoint = self.run_dir / "checkpoints/checkpoint_00000002.msgpack"
        self.assertTrue(first_checkpoint.is_file())
        restored, restored_path = load_latest_checkpoint(
            first, self.run_dir / "checkpoints"
        )
        self.assertEqual(restored_path, first_checkpoint)
        self.assertEqual(restored.epoch, first.epoch)
        self.assertEqual(restored.batch_in_epoch, first.batch_in_epoch)
        for expected, actual in zip(
            jax.tree.leaves(first), jax.tree.leaves(restored)
        ):
            np.testing.assert_array_equal(expected, actual)
        first_samples = np.load(self.run_dir / "samples/samples_00000002.npy")
        self.assertEqual(first_samples.shape, (2, 4, 4, 1))
        self.assertTrue(np.isfinite(first_samples).all())

        corrupt = self.run_dir / "checkpoints/checkpoint_99999999.msgpack"
        corrupt.write_bytes(b"not a checkpoint")
        second = self.run_until(4)
        self.assertEqual(int(second.state.step), 4)
        self.assertTrue(
            (self.run_dir / "checkpoints/checkpoint_00000004.msgpack").is_file()
        )
        self.assertTrue(
            (self.run_dir / "samples/samples_00000004.npy").is_file()
        )

        records = [
            json.loads(line)
            for line in (self.run_dir / "train.jsonl").read_text().splitlines()
        ]
        training_steps = [
            record["step"] for record in records if record["event"] == "train"
        ]
        self.assertEqual(training_steps, [1, 2, 3, 4])
        training_positions = [
            (record["epoch"], record["batch_in_epoch"])
            for record in records
            if record["event"] == "train"
        ]
        self.assertEqual(training_positions, [(0, 1), (0, 2), (0, 3), (1, 1)])
        resume_records = [
            record for record in records if record["event"] == "resume"
        ]
        self.assertEqual(len(resume_records), 1)
        self.assertEqual(resume_records[0]["checkpoint"], str(first_checkpoint))
        self.assertFalse(list(self.run_dir.rglob("*.tmp")))

    def test_run_configuration_must_match(self):
        path = self.run_dir / "config.json"
        ensure_run_config(path, {"base_channels": 8, "seed": 0})
        ensure_run_config(path, {"base_channels": 8, "seed": 0})
        with self.assertRaises(ValueError):
            ensure_run_config(path, {"base_channels": 16, "seed": 0})
        self.assertFalse(list(self.run_dir.rglob("*.tmp")))

    def test_training_settings_must_be_positive(self):
        with self.assertRaises(ValueError):
            run_training(
                TinyNoiseModel(),
                make_linear_schedule(2),
                np.zeros((1, 4, 4, 1), dtype=np.uint8),
                self.run_dir,
                batch_size=0,
                learning_rate=1e-3,
                max_steps=1,
                checkpoint_every=1,
                log_every=1,
                sample_every=1,
                num_samples=1,
            )


if __name__ == "__main__":
    unittest.main()
