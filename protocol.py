"""
Holographic Image Transmission Protocol
========================================

Compressive-sensing / single-pixel-camera style image transmission.

Sender and receiver share a random seed.  For every measurement the shared
PRNG produces a random ±1 mask the same size as the image.  The sender
transmits ``dot(mask, image)`` as a single analog scalar.  The receiver
reconstructs the image by accumulating ``measurement * mask``.

A "sync epoch" resets the PRNG every *sync_interval* measurements so that
sender and receiver stay in lock-step even if a few samples are lost.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_mask(rng: np.random.Generator, num_pixels: int) -> np.ndarray:
    """Return a flat ±1 Rademacher vector of length *num_pixels*."""
    # 0 or 1 → mapped to -1 or +1
    return rng.integers(0, 2, size=num_pixels, dtype=np.int8) * 2 - 1


# ---------------------------------------------------------------------------
# Transmitter
# ---------------------------------------------------------------------------

class Transmitter:
    """Encodes an image into a stream of scalar measurements.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image as a 2-D float array (H, W) with values in [0, 255].
    seed : int
        Shared random seed.
    sync_interval : int
        Number of measurements per sync epoch.  The PRNG is re-seeded at the
        start of each epoch so both sides stay synchronised.
    """

    def __init__(self, image: np.ndarray, seed: int = 42,
                 sync_interval: int = 1000):
        self.image_flat = image.ravel().astype(np.float64)
        self.num_pixels = self.image_flat.size
        self.seed = seed
        self.sync_interval = sync_interval
        self._epoch = 0
        self._idx_in_epoch = 0
        self._rng = np.random.default_rng(self._epoch_seed())

    # ---- internal helpers ------------------------------------------------

    def _epoch_seed(self) -> int:
        """Deterministic seed for the current epoch."""
        return self.seed + self._epoch * 997  # offset by a prime

    def _advance(self) -> None:
        """Move to the next measurement, resetting the PRNG at epoch
        boundaries."""
        self._idx_in_epoch += 1
        if self._idx_in_epoch >= self.sync_interval:
            self._epoch += 1
            self._idx_in_epoch = 0
            self._rng = np.random.default_rng(self._epoch_seed())

    # ---- public API ------------------------------------------------------

    def next_measurement(self) -> float:
        """Generate the next measurement (dot product)."""
        mask = generate_mask(self._rng, self.num_pixels)
        value = float(np.dot(mask.astype(np.float64), self.image_flat))
        self._advance()
        return value

    def transmit(self, num_measurements: int) -> np.ndarray:
        """Return an array of *num_measurements* consecutive measurements."""
        out = np.empty(num_measurements, dtype=np.float64)
        for i in range(num_measurements):
            out[i] = self.next_measurement()
        return out


# ---------------------------------------------------------------------------
# Channel model
# ---------------------------------------------------------------------------

def add_channel_noise(signal: np.ndarray, snr_db: float,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise to *signal* at the given SNR (in dB).

    SNR is defined as  10 * log10(signal_power / noise_power).
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class Receiver:
    """Reconstructs an image from scalar measurements using the shared PRNG.

    Parameters
    ----------
    seed : int
        Must match the transmitter's seed.
    width, height : int
        Image dimensions (640×480 by default).
    sync_interval : int
        Must match the transmitter's sync_interval.
    """

    def __init__(self, seed: int = 42, width: int = 640, height: int = 480,
                 sync_interval: int = 1000):
        self.width = width
        self.height = height
        self.num_pixels = width * height
        self.seed = seed
        self.sync_interval = sync_interval
        self._epoch = 0
        self._idx_in_epoch = 0
        self._rng = np.random.default_rng(self._epoch_seed())
        # Accumulator for the reconstructed image
        self._accumulator = np.zeros(self.num_pixels, dtype=np.float64)
        self._num_received = 0

    def _epoch_seed(self) -> int:
        return self.seed + self._epoch * 997

    def _advance(self) -> None:
        self._idx_in_epoch += 1
        if self._idx_in_epoch >= self.sync_interval:
            self._epoch += 1
            self._idx_in_epoch = 0
            self._rng = np.random.default_rng(self._epoch_seed())

    def receive(self, measurement: float) -> None:
        """Process a single incoming measurement."""
        mask = generate_mask(self._rng, self.num_pixels)
        self._accumulator += measurement * mask.astype(np.float64)
        self._num_received += 1
        self._advance()

    def receive_batch(self, measurements: np.ndarray) -> None:
        """Process a batch of measurements."""
        for m in measurements:
            self.receive(float(m))

    def get_image(self) -> np.ndarray:
        """Return the current estimate as a uint8 (H, W) image.

        The accumulator is divided by the number of measurements received,
        shifted and scaled to [0, 255], then clipped.
        """
        if self._num_received == 0:
            return np.zeros((self.height, self.width), dtype=np.uint8)

        est = self._accumulator / self._num_received
        # Normalise to 0-255 range
        mn, mx = est.min(), est.max()
        if mx - mn < 1e-12:
            normed = np.zeros_like(est)
        else:
            normed = (est - mn) / (mx - mn) * 255.0
        return normed.reshape(self.height, self.width).astype(np.uint8)

    @property
    def num_received(self) -> int:
        return self._num_received


# ---------------------------------------------------------------------------
# Convenience: run the full encode→channel→decode pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image: np.ndarray, num_measurements: int,
                 seed: int = 42, sync_interval: int = 1000,
                 snr_db: float | None = None) -> np.ndarray:
    """Encode *image*, optionally add noise, decode, return reconstruction.

    Uses the fast vectorised path by default.
    """
    return run_pipeline_fast(image, num_measurements, seed=seed,
                             sync_interval=sync_interval, snr_db=snr_db)


def run_pipeline_fast(image: np.ndarray, num_measurements: int,
                      seed: int = 42, sync_interval: int = 1000,
                      snr_db: float | None = None,
                      batch_size: int = 500) -> np.ndarray:
    """Vectorised encode→channel→decode pipeline.

    Uses byte-level RNG + np.unpackbits for ~8× faster mask generation,
    and float32 arithmetic to halve memory bandwidth.
    """
    h, w = image.shape[:2]
    num_pixels = h * w
    # Round up to next multiple of 8 for bit-unpacking
    padded = ((num_pixels + 7) // 8) * 8
    bytes_per_mask = padded // 8

    image_flat = image.ravel().astype(np.float32)
    accumulator = np.zeros(num_pixels, dtype=np.float64)

    remaining = num_measurements
    epoch = 0
    idx_in_epoch = 0

    def epoch_seed(ep: int) -> int:
        return seed + ep * 997

    rng = np.random.default_rng(epoch_seed(0))

    while remaining > 0:
        left_in_epoch = sync_interval - idx_in_epoch
        chunk = min(remaining, left_in_epoch, batch_size)

        # Generate random bytes and unpack to bits — 8× fewer RNG calls
        raw = rng.bytes(chunk * bytes_per_mask)
        bits = np.unpackbits(
            np.frombuffer(raw, dtype=np.uint8)
        ).reshape(chunk, padded)[:, :num_pixels]
        # Map 0/1 → -1/+1 as float32
        masks = bits.astype(np.float32) * 2 - 1

        # Encode: measurements = masks @ image  →  (chunk,)
        measurements = masks @ image_flat

        # Add channel noise if requested
        if snr_db is not None:
            measurements = add_channel_noise(
                measurements.astype(np.float64), snr_db
            ).astype(np.float32)

        # Decode: accumulator += masks.T @ measurements
        accumulator += (masks.T @ measurements).astype(np.float64)

        remaining -= chunk
        idx_in_epoch += chunk
        if idx_in_epoch >= sync_interval:
            epoch += 1
            idx_in_epoch = 0
            rng = np.random.default_rng(epoch_seed(epoch))

    # Normalise to [0, 255]
    est = accumulator / num_measurements
    mn, mx = est.min(), est.max()
    if mx - mn < 1e-12:
        normed = np.zeros_like(est)
    else:
        normed = (est - mn) / (mx - mn) * 255.0
    return normed.reshape(h, w).astype(np.uint8)

