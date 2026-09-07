"""MNIST training images, labels, and reproducibly shuffled JAX batches."""

import gzip
from pathlib import Path
import shutil
import struct
from urllib.request import urlopen

import jax
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MNIST_IMAGE_URL = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/"
    "train-images-idx3-ubyte.gz"
)
MNIST_LABEL_URL = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/"
    "train-labels-idx1-ubyte.gz"
)


def _read_images(path):
    """Read the big-endian IDX header and uncompressed pixel bytes."""
    with gzip.open(path, "rb") as source:
        header = source.read(16)
        if len(header) != 16:
            raise ValueError("Incomplete MNIST image header")
        magic, count, height, width = struct.unpack(">IIII", header)
        if magic != 2051 or count < 1 or (height, width) != (28, 28):
            raise ValueError("Invalid MNIST image header")
        pixels = np.frombuffer(source.read(), dtype=np.uint8)
    if pixels.size != count * height * width:
        raise ValueError("MNIST image count does not match the pixel data")
    return pixels.reshape(count, height, width, 1)


def _read_labels(path):
    """Read the big-endian IDX header and digit labels."""
    with gzip.open(path, "rb") as source:
        header = source.read(8)
        if len(header) != 8:
            raise ValueError("Incomplete MNIST label header")
        magic, count = struct.unpack(">II", header)
        if magic != 2049 or count < 1:
            raise ValueError("Invalid MNIST label header")
        labels = np.frombuffer(source.read(), dtype=np.uint8)
    if labels.size != count or np.any(labels > 9):
        raise ValueError("Invalid MNIST label data")
    return labels


def _download(filename, url, reader):
    """Download and validate one MNIST IDX file unless it is already cached."""
    path = DATA_DIR / filename
    if path.exists():
        return reader(path)

    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        values = reader(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return values


def load_mnist():
    """Download once; return aligned host uint8 training images and labels."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images = _download("train-images-idx3-ubyte.gz", MNIST_IMAGE_URL, _read_images)
    labels = _download("train-labels-idx1-ubyte.gz", MNIST_LABEL_URL, _read_labels)
    if len(images) != len(labels):
        raise ValueError("MNIST image and label counts do not match")
    return images, labels


def one_hot(labels):
    """Convert MNIST digit labels to ten-dimensional float32 covariates."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels > 9):
        raise ValueError("labels must be a one-dimensional array of digits 0 through 9")
    return np.eye(10, dtype=np.float32)[labels]


def batches(images, batch_size, *, seed):
    """Yield one shuffled epoch, including its final partial batch.

    Use a different seed for each epoch. Only each batch is normalized and sent
    to JAX's default device, so the full dataset does not occupy GPU memory.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    order = np.random.default_rng(seed).permutation(len(images))
    for start in range(0, len(images), batch_size):
        pixels = images[order[start : start + batch_size]]
        clean_images = pixels.astype(np.float32) / 127.5 - 1.0
        yield jax.device_put(clean_images)


def conditional_batches(images, labels, batch_size, *, seed):
    """Yield aligned image and one-hot covariate batches for one shuffled epoch."""
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    order = np.random.default_rng(seed).permutation(len(images))
    for start in range(0, len(images), batch_size):
        indices = order[start : start + batch_size]
        clean_images = images[indices].astype(np.float32) / 127.5 - 1.0
        covariates = one_hot(labels[indices])
        yield jax.device_put(clean_images), jax.device_put(covariates)
