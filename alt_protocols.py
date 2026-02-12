"""
Alternative transmission protocols for holographic image transmission.

Each protocol provides TX (transmitter) and RX (receiver) classes
with the same interface as HoloTx/HoloRx:
  - TX: __init__(seed, width, height, image_array), generate(count) -> float32[]
  - RX: __init__(seed, width, height), accumulate(measurements), get_image() -> uint8[H,W]

Line and Block protocols use a C shared library (libalt_fast.so) for performance.
Falls back to pure Python if the library is not found.
"""

import numpy as np
import ctypes
import os

# ============================================================
# Load C library (optional — falls back to Python)
# ============================================================
_lib = None
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libalt_fast.so")
try:
    _lib = ctypes.CDLL(_lib_path)

    # Line TX
    _lib.line_tx_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,  # img, w, h
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # x0s, y0s
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # x1s, y1s
        ctypes.POINTER(ctypes.c_float), ctypes.c_int                 # output, count
    ]
    _lib.line_tx_batch.restype = None

    # Line RX
    _lib.line_rx_batch.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),  # accum, counts
        ctypes.c_int, ctypes.c_int,                                      # w, h
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),     # x0s, y0s
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),     # x1s, y1s
        ctypes.POINTER(ctypes.c_double), ctypes.c_int                    # measurements, count
    ]
    _lib.line_rx_batch.restype = None

    # Block TX
    _lib.block_tx_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,  # img, w, h
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # bws, bhs
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),  # x0s, y0s
        ctypes.POINTER(ctypes.c_float), ctypes.c_int                 # output, count
    ]
    _lib.block_tx_batch.restype = None

    # Block RX
    _lib.block_rx_batch.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),  # accum, counts
        ctypes.c_int, ctypes.c_int,                                      # w, h
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),     # bws, bhs
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),     # x0s, y0s
        ctypes.POINTER(ctypes.c_double), ctypes.c_int                    # measurements, count
    ]
    _lib.block_rx_batch.restype = None

    print("[INFO] Loaded libalt_fast.so — Line/Block protocols use C+OpenMP")
except OSError:
    print("[WARN] libalt_fast.so not found — Line/Block protocols will use slow Python fallback")
    _lib = None


def _ptr(arr, ctype):
    """Get ctypes pointer from numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


# ============================================================
# Protocol 1: Random Pixel Sampling (pure numpy — already fast)
# ============================================================

class PixelTx:
    """Transmitter: samples random individual pixels."""
    def __init__(self, seed, width, height, image_array):
        self.width = width
        self.height = height
        self.img = image_array.astype(np.float32).flatten()
        self.rng = np.random.default_rng(seed)
        self.npix = width * height

    def generate(self, count):
        indices = self.rng.integers(0, self.npix, size=count)
        return self.img[indices]

    def close(self):
        pass


class PixelRx:
    """Receiver: accumulates pixel samples and averages repeated pixels."""
    def __init__(self, seed, width, height):
        self.width = width
        self.height = height
        self.npix = width * height
        self.rng = np.random.default_rng(seed)
        self.accum = np.zeros(self.npix, dtype=np.float64)
        self.counts = np.zeros(self.npix, dtype=np.int32)
        self.measurements_count = 0

    def accumulate(self, measurements):
        meas = np.asarray(measurements, dtype=np.float64)
        count = len(meas)
        indices = self.rng.integers(0, self.npix, size=count)
        np.add.at(self.accum, indices, meas)
        np.add.at(self.counts, indices, 1)
        self.measurements_count += count

    def get_image(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = np.where(self.counts > 0, self.accum / self.counts, 128.0)
        return np.clip(avg, 0, 255).astype(np.uint8).reshape((self.height, self.width))

    def close(self):
        pass


# ============================================================
# Protocol 2: Random Line Sampling (C-accelerated)
# ============================================================

def _bresenham_line(x0, y0, x1, y1):
    """Pure Python fallback for Bresenham's line algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


class LineTx:
    """Transmitter: samples average brightness along random lines."""
    def __init__(self, seed, width, height, image_array):
        self.width = width
        self.height = height
        self.img = image_array.astype(np.float32).flatten()  # row-major
        self.img_2d = self.img.reshape((height, width))
        self.rng = np.random.default_rng(seed)

    def generate(self, count):
        # Generate all random endpoints in batch (numpy)
        x0s = self.rng.integers(0, self.width, size=count).astype(np.int32)
        x1s = self.rng.integers(0, self.width, size=count).astype(np.int32)
        y0s = self.rng.integers(0, self.height, size=count).astype(np.int32)
        y1s = self.rng.integers(0, self.height, size=count).astype(np.int32)

        out = np.zeros(count, dtype=np.float32)

        if _lib is not None:
            # C path (OpenMP parallel)
            _lib.line_tx_batch(
                _ptr(self.img, ctypes.c_float), self.width, self.height,
                _ptr(x0s, ctypes.c_int), _ptr(y0s, ctypes.c_int),
                _ptr(x1s, ctypes.c_int), _ptr(y1s, ctypes.c_int),
                _ptr(out, ctypes.c_float), count
            )
        else:
            # Python fallback
            for i in range(count):
                pts = _bresenham_line(x0s[i], y0s[i], x1s[i], y1s[i])
                vals = [self.img_2d[y, x] for x, y in pts]
                out[i] = np.mean(vals)

        return out

    def close(self):
        pass


class LineRx:
    """Receiver: distributes line measurements to all pixels on each line."""
    def __init__(self, seed, width, height):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.accum = np.zeros(height * width, dtype=np.float64)
        self.counts = np.zeros(height * width, dtype=np.int32)
        self.measurements_count = 0

    def accumulate(self, measurements):
        meas = np.asarray(measurements, dtype=np.float64)
        count = len(meas)

        x0s = self.rng.integers(0, self.width, size=count).astype(np.int32)
        x1s = self.rng.integers(0, self.width, size=count).astype(np.int32)
        y0s = self.rng.integers(0, self.height, size=count).astype(np.int32)
        y1s = self.rng.integers(0, self.height, size=count).astype(np.int32)

        if _lib is not None:
            # C path
            _lib.line_rx_batch(
                _ptr(self.accum, ctypes.c_double), _ptr(self.counts, ctypes.c_int),
                self.width, self.height,
                _ptr(x0s, ctypes.c_int), _ptr(y0s, ctypes.c_int),
                _ptr(x1s, ctypes.c_int), _ptr(y1s, ctypes.c_int),
                _ptr(meas, ctypes.c_double), count
            )
        else:
            # Python fallback
            accum_2d = self.accum.reshape((self.height, self.width))
            counts_2d = self.counts.reshape((self.height, self.width))
            for i in range(count):
                pts = _bresenham_line(x0s[i], y0s[i], x1s[i], y1s[i])
                val = meas[i]
                for x, y in pts:
                    accum_2d[y, x] += val
                    counts_2d[y, x] += 1

        self.measurements_count += count

    def get_image(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = np.where(self.counts > 0, self.accum / self.counts, 128.0)
        return np.clip(avg, 0, 255).astype(np.uint8).reshape((self.height, self.width))

    def close(self):
        pass


# ============================================================
# Protocol 3: Random Block Sampling (C-accelerated)
# ============================================================

class BlockTx:
    """Transmitter: samples average brightness of random rectangles."""
    def __init__(self, seed, width, height, image_array):
        self.width = width
        self.height = height
        self.img = image_array.astype(np.float32).flatten()
        self.img_2d = self.img.reshape((height, width))
        self.rng = np.random.default_rng(seed)

    def generate(self, count):
        bws = self.rng.integers(4, min(65, self.width + 1), size=count).astype(np.int32)
        bhs = self.rng.integers(4, min(65, self.height + 1), size=count).astype(np.int32)
        x0s = np.array([self.rng.integers(0, self.width - bw + 1) for bw in bws], dtype=np.int32)
        y0s = np.array([self.rng.integers(0, self.height - bh + 1) for bh in bhs], dtype=np.int32)

        out = np.zeros(count, dtype=np.float32)

        if _lib is not None:
            _lib.block_tx_batch(
                _ptr(self.img, ctypes.c_float), self.width, self.height,
                _ptr(bws, ctypes.c_int), _ptr(bhs, ctypes.c_int),
                _ptr(x0s, ctypes.c_int), _ptr(y0s, ctypes.c_int),
                _ptr(out, ctypes.c_float), count
            )
        else:
            for i in range(count):
                out[i] = np.mean(self.img_2d[y0s[i]:y0s[i]+bhs[i], x0s[i]:x0s[i]+bws[i]])

        return out

    def close(self):
        pass


class BlockRx:
    """Receiver: distributes block measurements to all pixels in each block."""
    def __init__(self, seed, width, height):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.accum = np.zeros(height * width, dtype=np.float64)
        self.counts = np.zeros(height * width, dtype=np.int32)
        self.measurements_count = 0

    def accumulate(self, measurements):
        meas = np.asarray(measurements, dtype=np.float64)
        count = len(meas)

        bws = self.rng.integers(4, min(65, self.width + 1), size=count).astype(np.int32)
        bhs = self.rng.integers(4, min(65, self.height + 1), size=count).astype(np.int32)
        x0s = np.array([self.rng.integers(0, self.width - bw + 1) for bw in bws], dtype=np.int32)
        y0s = np.array([self.rng.integers(0, self.height - bh + 1) for bh in bhs], dtype=np.int32)

        if _lib is not None:
            _lib.block_rx_batch(
                _ptr(self.accum, ctypes.c_double), _ptr(self.counts, ctypes.c_int),
                self.width, self.height,
                _ptr(bws, ctypes.c_int), _ptr(bhs, ctypes.c_int),
                _ptr(x0s, ctypes.c_int), _ptr(y0s, ctypes.c_int),
                _ptr(meas, ctypes.c_double), count
            )
        else:
            accum_2d = self.accum.reshape((self.height, self.width))
            counts_2d = self.counts.reshape((self.height, self.width))
            for i in range(count):
                accum_2d[y0s[i]:y0s[i]+bhs[i], x0s[i]:x0s[i]+bws[i]] += meas[i]
                counts_2d[y0s[i]:y0s[i]+bhs[i], x0s[i]:x0s[i]+bws[i]] += 1

        self.measurements_count += count

    def get_image(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = np.where(self.counts > 0, self.accum / self.counts, 128.0)
        return np.clip(avg, 0, 255).astype(np.uint8).reshape((self.height, self.width))

    def close(self):
        pass


# ============================================================
# Registry
# ============================================================
PROTOCOLS = {
    "Random Pixel": (PixelTx, PixelRx),
    "Random Line": (LineTx, LineRx),
    "Random Block": (BlockTx, BlockRx),
}
