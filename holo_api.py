
import ctypes
import os
import numpy as np

# --------------------------------------------------------------------------
# GPU backend (CUDA)
# --------------------------------------------------------------------------
_lib = None
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libholo.so")
if os.path.exists(_lib_path):
    try:
        _lib = ctypes.CDLL(_lib_path)

        class TxState(ctypes.Structure): pass
        class RxState(ctypes.Structure): pass

        _lib.tx_create.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        _lib.tx_create.restype = ctypes.POINTER(TxState)
        _lib.tx_destroy.argtypes = [ctypes.POINTER(TxState)]
        _lib.tx_destroy.restype = None
        _lib.tx_generate.argtypes = [ctypes.POINTER(TxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.tx_generate.restype = None

        _lib.rx_create.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        _lib.rx_create.restype = ctypes.POINTER(RxState)
        _lib.rx_destroy.argtypes = [ctypes.POINTER(RxState)]
        _lib.rx_destroy.restype = None
        _lib.rx_accumulate.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.rx_accumulate.restype = None
        _lib.rx_get_image.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        _lib.rx_get_image.restype = None
        
        _lib.rx_solve_ls.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]
        _lib.rx_solve_ls.restype = None
        _lib.rx_get_ls_image.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_uint8)]
        _lib.rx_get_ls_image.restype = None
    except OSError as e:
        print(f"[WARN] Failed to load libholo.so: {e}")
        _lib = None


class HoloTx:
    """GPU Holographic Transmitter (CUDA)"""
    def __init__(self, seed, width, height, image_array):
        if _lib is None:
            raise RuntimeError("libholo.so not loaded")
        self.ptr = None
        self.width = width
        self.height = height
        img_float = image_array.astype(np.float32).flatten()
        if len(img_float) != width * height:
            raise ValueError(f"Image size mismatch: expected {width}x{height}={width*height}, got {len(img_float)}")
        self.ptr = _lib.tx_create(seed, width, height, img_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if not self.ptr:
            raise RuntimeError("GPU TX init failed (cuBLAS/cuRAND/CUDA error). Check stderr for details.")

    def generate(self, count):
        out = np.zeros(count, dtype=np.float32)
        _lib.tx_generate(self.ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
        return out

    def close(self):
        if self.ptr:
            _lib.tx_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()


class HoloRx:
    """GPU Holographic Receiver (CUDA)"""
    def __init__(self, seed, width, height):
        if _lib is None:
            raise RuntimeError("libholo.so not loaded")
        self.ptr = None
        self.seed = seed
        self.width = width
        self.height = height
        self.ptr = _lib.rx_create(seed, width, height)
        if not self.ptr:
            raise RuntimeError("GPU RX init failed (cuBLAS/cuRAND/CUDA error). Check stderr for details.")
        self.measurements_count = 0

    def accumulate(self, measurements):
        meas_float = measurements.astype(np.float32)
        count = len(meas_float)
        _lib.rx_accumulate(self.ptr, meas_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
        self.measurements_count += count

    def get_image(self):
        out = np.zeros(self.width * self.height, dtype=np.uint8)
        _lib.rx_get_image(self.ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.measurements_count)
        return out.reshape((self.height, self.width))

    def solve_ls(self, measurements, iterations=20, tol=1e-6):
        """Solve for image using CGLS (Least Squares). Replaces current accumulation state."""
        meas_float = measurements.astype(np.float32)
        count = len(meas_float)
        # Note: solve_ls OVERWRITES the accumulator with the LS result
        _lib.rx_solve_ls(self.ptr, meas_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count, iterations, tol)
        
        out = np.zeros(self.width * self.height, dtype=np.uint8)
        _lib.rx_get_ls_image(self.ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape((self.height, self.width))

    def reset(self):
        self.close()
        self.ptr = _lib.rx_create(self.seed, self.width, self.height)
        if not self.ptr:
            raise RuntimeError("GPU RX re-init failed (cuBLAS/cuRAND/CUDA error). Check stderr for details.")
        self.measurements_count = 0

    def close(self):
        if self.ptr:
            _lib.rx_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()


# --------------------------------------------------------------------------
# CPU fallback (pure Python via protocol.py)
# --------------------------------------------------------------------------
from protocol import Transmitter, Receiver


class HoloTxCPU:
    """CPU Holographic Transmitter — fallback when GPU is unavailable."""
    def __init__(self, seed, width, height, image_array):
        self.ptr = "cpu"  # non-None so close() guards work
        self.width = width
        self.height = height
        if image_array.size != width * height:
            raise ValueError(f"Image size mismatch: expected {width}x{height}={width*height}, got {image_array.size}")
        self._tx = Transmitter(image_array.reshape(height, width), seed=seed)

    def generate(self, count):
        return self._tx.transmit(count).astype(np.float32)

    def close(self):
        self.ptr = None

    def __del__(self):
        self.close()


class HoloRxCPU:
    """CPU Holographic Receiver — fallback when GPU is unavailable."""
    def __init__(self, seed, width, height):
        self.ptr = "cpu"
        self.seed = seed
        self.width = width
        self.height = height
        self._rx = Receiver(seed=seed, width=width, height=height)
        self.measurements_count = 0

    def accumulate(self, measurements):
        self._rx.receive_batch(measurements.astype(np.float64))
        self.measurements_count += len(measurements)

    def get_image(self):
        return self._rx.get_image()

    def reset(self):
        self._rx = Receiver(seed=self.seed, width=self.width, height=self.height)
        self.measurements_count = 0

    def close(self):
        self.ptr = None

    def __del__(self):
        self.close()


# --------------------------------------------------------------------------
# GPU availability probe
# --------------------------------------------------------------------------
GPU_AVAILABLE = False
if _lib is not None:
    try:
        _probe = HoloRx(999, 4, 4)
        _probe.close()
        GPU_AVAILABLE = True
    except RuntimeError:
        GPU_AVAILABLE = False

if GPU_AVAILABLE:
    print("[INFO] GPU holographic engine available (CUDA)")
else:
    print("[INFO] GPU unavailable — using CPU holographic engine (slower)")
