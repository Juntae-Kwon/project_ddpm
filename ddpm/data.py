"""MNIST training images and simple, reproducibly shuffled JAX batches."""

import gzip
from pathlib import Path
import shutil
import struct
from urllib.request import urlopen

import jax
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MNIST_URL = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/"
    "train-images-idx3-ubyte.gz"
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


def load_mnist():
    """Download once under the project root; return host uint8 training images."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "train-images-idx3-ubyte.gz"
    if path.exists():
        return _read_images(path)

    # An interrupted download must not become the cached dataset.
    temporary = path.with_suffix(".gz.part")
    try:
        with urlopen(MNIST_URL, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        images = _read_images(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return images


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
