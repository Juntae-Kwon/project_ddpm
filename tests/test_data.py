"""Offline tests with tiny IDX fixtures; no MNIST download is needed."""

import gzip
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from ddpm import data


class DataTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".cache" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "images.gz"

    def write_images(self, header, pixels):
        with gzip.open(self.path, "wb") as target:
            target.write(header + pixels)

    def test_read_images(self):
        pixels = np.arange(2 * 28 * 28).astype(np.uint8)
        self.write_images(struct.pack(">IIII", 2051, 2, 28, 28), pixels.tobytes())
        images = data._read_images(self.path)
        self.assertEqual(images.shape, (2, 28, 28, 1))
        self.assertEqual(images.dtype, np.uint8)
        np.testing.assert_array_equal(images.ravel(), pixels)

    def test_invalid_headers_and_lengths(self):
        for header, pixels in (
            (b"short", b""),
            (struct.pack(">IIII", 2049, 1, 28, 28), bytes(784)),
            (struct.pack(">IIII", 2051, 1, 27, 28), bytes(756)),
            (struct.pack(">IIII", 2051, 1, 28, 28), bytes(783)),
            (struct.pack(">IIII", 2051, 1, 28, 28), bytes(785)),
        ):
            with self.subTest(header=header, length=len(pixels)):
                self.write_images(header, pixels)
                with self.assertRaises(ValueError):
                    data._read_images(self.path)

    def test_download_and_cache(self):
        payload = gzip.compress(struct.pack(">IIII", 2051, 1, 28, 28) + bytes(784))
        with patch.object(data, "DATA_DIR", Path(self.directory.name)), patch.object(
            data, "urlopen", return_value=io.BytesIO(payload)
        ) as download:
            first = data.load_mnist()
            second = data.load_mnist()
            download.assert_called_once()
        np.testing.assert_array_equal(first, second)
        self.assertFalse(list(Path(self.directory.name).glob("*.part")))

    def test_invalid_download_is_not_cached(self):
        with patch.object(data, "DATA_DIR", Path(self.directory.name)), patch.object(
            data, "urlopen", return_value=io.BytesIO(gzip.compress(b"short"))
        ):
            with self.assertRaises(ValueError):
                data.load_mnist()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_batches(self):
        values = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        images = np.broadcast_to(values[:, None, None, None], (5, 28, 28, 1))
        result = list(data.batches(images, 2, seed=7))
        self.assertEqual([batch.shape for batch in result],
                         [(2, 28, 28, 1), (2, 28, 28, 1), (1, 28, 28, 1)])
        for batch in result:
            self.assertEqual(batch.dtype, np.float32)
            self.assertEqual(batch.devices(), jax.device_put(0).devices())
        actual = np.concatenate([np.asarray(batch) for batch in result])
        expected = images[np.random.default_rng(7).permutation(5)].astype(np.float32)
        np.testing.assert_array_equal(actual, expected / 127.5 - 1.0)
        repeated = np.concatenate(list(data.batches(images, 2, seed=7)))
        np.testing.assert_array_equal(actual, repeated)
        self.assertEqual(actual.min(), -1.0)
        self.assertEqual(actual.max(), 1.0)
        np.testing.assert_array_equal(images[:, 0, 0, 0], values)

    def test_invalid_batch_size(self):
        for size in (0, -1):
            with self.assertRaises(ValueError):
                list(data.batches(np.zeros((1, 28, 28, 1), dtype=np.uint8), size, seed=0))


if __name__ == "__main__":
    unittest.main()
