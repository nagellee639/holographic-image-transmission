
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
        class SolverState(ctypes.Structure): pass

        # TX
        _lib.tx_create.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        _lib.tx_create.restype = ctypes.POINTER(TxState)
        _lib.tx_destroy.argtypes = [ctypes.POINTER(TxState)]
        _lib.tx_destroy.restype = None
        _lib.tx_generate.argtypes = [ctypes.POINTER(TxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.tx_generate.restype = None

        _lib.tx_generate_fast.argtypes = [ctypes.POINTER(TxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.tx_generate_fast.restype = None

        # RX (Accumulator/Legacy)
        _lib.rx_create.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        _lib.rx_create.restype = ctypes.POINTER(RxState)
        _lib.rx_destroy.argtypes = [ctypes.POINTER(RxState)]
        _lib.rx_destroy.restype = None
        _lib.rx_accumulate.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.rx_accumulate.restype = None
        _lib.rx_get_image.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        _lib.rx_get_image.restype = None
        
        # SOLVER (Stateful)
        _lib.solver_create_ls.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.solver_create_ls.restype = ctypes.POINTER(SolverState)

        _lib.solver_create_ls_fast.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _lib.solver_create_ls_fast.restype = ctypes.POINTER(SolverState)
        
        _lib.solver_create_l1.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
        _lib.solver_create_l1.restype = ctypes.POINTER(SolverState)

        _lib.solver_create_l1_fast.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
        _lib.solver_create_l1_fast.restype = ctypes.POINTER(SolverState)
        
        _lib.solver_step.argtypes = [ctypes.POINTER(SolverState)]
        _lib.solver_step.restype = None
        
        _lib.solver_project_histogram.argtypes = [ctypes.POINTER(SolverState), ctypes.POINTER(ctypes.c_int)]
        _lib.solver_project_histogram.restype = None
        
        _lib.solver_get_image.argtypes = [ctypes.POINTER(SolverState), ctypes.POINTER(ctypes.c_uint8)]
        _lib.solver_get_image.restype = None
        
        _lib.solver_destroy.argtypes = [ctypes.POINTER(SolverState)]
        _lib.solver_destroy.restype = None
        
        _lib.solver_set_meas_offset.argtypes = [ctypes.POINTER(SolverState), ctypes.c_int]
        _lib.solver_set_meas_offset.restype = None
        
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
            raise RuntimeError("GPU TX init failed")
        self.generated_count = 0

    def generate(self, count, fast=False):
        """Generate `count` measurements. Returns (measurements, start_offset)."""
        out = np.zeros(count, dtype=np.float32)
        start_offset = self.generated_count
        if fast:
             _lib.tx_generate_fast(self.ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
        else:
             _lib.tx_generate(self.ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
        
        self.generated_count += count
        return out, start_offset

    def close(self):
        if self.ptr:
            _lib.tx_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()


class HoloRx:
    """GPU Holographic Receiver (Accumulator + Solver Factory)"""
    def __init__(self, seed, width, height):
        if _lib is None:
            raise RuntimeError("libholo.so not loaded")
        self.ptr = None
        self.seed = seed
        self.width = width
        self.height = height
        self.ptr = _lib.rx_create(seed, width, height)
        if not self.ptr:
            raise RuntimeError("GPU RX init failed")
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

    def create_solver_ls(self, measurements, fast=False):
        """Create a stateful Least Squares (CGLS) solver."""
        mode = "ls_fast" if fast else "ls"
        return HoloSolver(self, measurements, mode=mode)
        
    def create_solver_l1(self, measurements, lambda_val=0.1, fast=False):
        """Create a stateful L1 (ISTA/FISTA) solver."""
        mode = "l1_fast" if fast else "l1"
        return HoloSolver(self, measurements, mode=mode, lambda_val=lambda_val)

    def reset(self):
        self.close()
        self.ptr = _lib.rx_create(self.seed, self.width, self.height)
        self.measurements_count = 0

    def close(self):
        if self.ptr:
            _lib.rx_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()
        
    # Legacy wrapper for 'solve_ls' (one-shot) - kept for backward compat if needed,
    # but strictly we can remove it if GUI is updated. 
    # Let's keep a blocking version for tests
    def solve_ls_blocking(self, measurements, iterations=15):
        s = self.create_solver_ls(measurements)
        for _ in range(iterations):
            s.step()
        return s.get_image()


class HoloSolver:
    """Stateful Solver Wrapper"""
    def __init__(self, parent_rx, measurements, mode="ls", lambda_val=0.1):
        self.width = parent_rx.width
        self.height = parent_rx.height
        meas_float = measurements.astype(np.float32)
        n_meas = len(meas_float)
        
        if mode == "ls":
            self.ptr = _lib.solver_create_ls(parent_rx.seed, self.width, self.height, 
                                             meas_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_meas)
        elif mode == "ls_fast":
            self.ptr = _lib.solver_create_ls_fast(parent_rx.seed, self.width, self.height, 
                                                  meas_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_meas)
        elif mode == "l1":
            self.ptr = _lib.solver_create_l1(parent_rx.seed, self.width, self.height, 
                                             meas_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_meas, lambda_val)
        elif mode == "l1_fast":
            self.ptr = _lib.solver_create_l1_fast(parent_rx.seed, self.width, self.height, 
                                                  meas_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_meas, lambda_val)
        else:
            raise ValueError(f"Unknown mode {mode}")
            
        if not self.ptr:
            raise RuntimeError("Solver init failed")
            
    def step(self):
        _lib.solver_step(self.ptr)
        
    def project_histogram(self, histogram):
        hist_int = histogram.astype(np.int32)
        if len(hist_int) != 256:
             raise ValueError("Histogram must size 256")
        _lib.solver_project_histogram(self.ptr, hist_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        
    def get_image(self):
        out = np.zeros(self.width * self.height, dtype=np.uint8)
        _lib.solver_get_image(self.ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape((self.height, self.width))
        
    def set_offset(self, offset):
        if self.ptr:
             _lib.solver_set_meas_offset(self.ptr, offset)
        
    def close(self):
        if self.ptr:
             _lib.solver_destroy(self.ptr)
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
        self.ptr = "cpu"
        self.width = width
        self.height = height
        if image_array.size != width * height:
            raise ValueError(f"Image size mismatch: expected {width}x{height}={width*height}, got {image_array.size}")
        self._tx = Transmitter(image_array.reshape(height, width), seed=seed)

    def generate(self, count):
        return self._tx.transmit(count).astype(np.float32)

    def close(self):
        self.ptr = None

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
        
    # CPU doesn't support stateful solver visualization yet
    # Implementation of create_solver_* would require refactoring protocol.py
    # For now, raise NotImplemented or return dummy
    def create_solver_ls(self, measurements):
        raise NotImplementedError("Stateful solver not implemented for CPU backend")

    def create_solver_l1(self, measurements, lambda_val=0.1):
        raise NotImplementedError("Stateful solver not implemented for CPU backend")

    def reset(self):
        self._rx = Receiver(seed=self.seed, width=self.width, height=self.height)
        self.measurements_count = 0

    def close(self):
        self.ptr = None


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
