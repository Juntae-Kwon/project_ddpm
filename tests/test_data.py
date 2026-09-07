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
        self.image_path = Path(self.directory.name) / "images.gz"
        self.label_path = Path(self.directory.name) / "labels.gz"

    def write_images(self, header, pixels):
        with gzip.open(self.image_path, "wb") as target:
            target.write(header + pixels)

    def test_read_images(self):
        pixels = np.arange(2 * 28 * 28).astype(np.uint8)
        self.write_images(struct.pack(">IIII", 2051, 2, 28, 28), pixels.tobytes())
        images = data._read_images(self.image_path)
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
                    data._read_images(self.image_path)

    def test_read_labels_and_one_hot(self):
        labels = np.array([0, 4, 9], dtype=np.uint8)
        with gzip.open(self.label_path, "wb") as target:
            target.write(struct.pack(">II", 2049, 3) + labels.tobytes())
        actual = data._read_labels(self.label_path)
        np.testing.assert_array_equal(actual, labels)
        covariates = data.one_hot(actual)
        self.assertEqual(covariates.shape, (3, 10))
        self.assertEqual(covariates.dtype, np.float32)
        np.testing.assert_array_equal(covariates.sum(axis=1), 1.0)
        np.testing.assert_array_equal(covariates.argmax(axis=1), labels)

        for invalid in (np.array([-1]), np.array([10]), np.zeros((1, 1))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                data.one_hot(invalid)

    def test_download_and_cache(self):
        image_payload = gzip.compress(
            struct.pack(">IIII", 2051, 1, 28, 28) + bytes(784)
        )
        label_payload = gzip.compress(struct.pack(">II", 2049, 1) + b"\x07")
        with patch.object(data, "DATA_DIR", Path(self.directory.name)), patch.object(
            data,
            "urlopen",
            side_effect=[io.BytesIO(image_payload), io.BytesIO(label_payload)],
        ) as download:
            first_images, first_labels = data.load_mnist()
            second_images, second_labels = data.load_mnist()
            self.assertEqual(download.call_count, 2)
        np.testing.assert_array_equal(first_images, second_images)
        np.testing.assert_array_equal(first_labels, second_labels)
        np.testing.assert_array_equal(first_labels, [7])
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

    def test_conditional_batches_keep_images_and_labels_aligned(self):
        labels = np.array([0, 2, 4, 6, 8], dtype=np.uint8)
        images = np.broadcast_to(
            labels[:, None, None, None], (5, 28, 28, 1)
        ).copy()
        result = list(data.conditional_batches(images, labels, 2, seed=3))
        self.assertEqual([images.shape[0] for images, _ in result], [2, 2, 1])
        for image_batch, covariate_batch in result:
            self.assertEqual(covariate_batch.shape, (image_batch.shape[0], 10))
            self.assertEqual(image_batch.devices(), covariate_batch.devices())
            recovered_pixels = np.rint((np.asarray(image_batch) + 1.0) * 127.5)
            recovered_labels = np.asarray(covariate_batch).argmax(axis=1)
            np.testing.assert_array_equal(recovered_pixels[:, 0, 0, 0], recovered_labels)

        with self.assertRaises(ValueError):
            list(data.conditional_batches(images, labels[:-1], 2, seed=0))

    def test_invalid_batch_size(self):
        for size in (0, -1):
            with self.assertRaises(ValueError):
                list(data.batches(np.zeros((1, 28, 28, 1), dtype=np.uint8), size, seed=0))


if __name__ == "__main__":
    unittest.main()
