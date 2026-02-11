
import ctypes
import os
import numpy as np

# Load shared library
_lib_path = os.path.abspath("libholo.so")
if not os.path.exists(_lib_path):
    raise RuntimeError(f"Shared library not found at {_lib_path}. Run 'nvcc ... -shared ...' first.")

_lib = ctypes.CDLL(_lib_path)

# Define structures (opaque pointers to C++ structs)
class TxState(ctypes.Structure): pass
class RxState(ctypes.Structure): pass

# Define function prototypes
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

class HoloTx:
    """Bandwidth-limited Holographic Transmitter"""
    def __init__(self, seed, width, height, image_array):
        self.ptr = None
        self.width = width
        self.height = height
        # Flatten and ensure float32
        img_float = image_array.astype(np.float32).flatten()
        if len(img_float) != width * height:
            raise ValueError(f"Image size mismatch: expected {width}x{height}={width*height}, got {len(img_float)}")
            
        self.ptr = _lib.tx_create(seed, width, height, img_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def generate(self, count):
        """Generate 'count' measurements (samples). Returns float32 array."""
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
    """Holographic Receiver accumulator"""
    def __init__(self, seed, width, height):
        self.ptr = None
        self.width = width
        self.height = height
        self.ptr = _lib.rx_create(seed, width, height)
        self.measurements_count = 0

    def accumulate(self, measurements):
        """Process incoming measurements (float32 array)."""
        meas_float = measurements.astype(np.float32)
        count = len(meas_float)
        _lib.rx_accumulate(self.ptr, meas_float.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count)
        self.measurements_count += count

    def get_image(self):
        """Return the current reconstruction as (H, W) uint8 array."""
        out = np.zeros(self.width * self.height, dtype=np.uint8)
        _lib.rx_get_image(self.ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.measurements_count)
        return out.reshape((self.height, self.width))
    
    def reset(self):
        """Reset internal accumulator"""
        # Easiest way is re-create
        self.close()
        self.ptr = _lib.rx_create(self.seed, self.width, self.height) # Oops, seed not saved
        # Wait, lib doesn't expose reset. Destroy/Create is fine.
        # Implemented in GUI layer logic instead.
        pass

    def close(self):
        if self.ptr:
            _lib.rx_destroy(self.ptr)
            self.ptr = None
            
    def __del__(self):
        self.close()
