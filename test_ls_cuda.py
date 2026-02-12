import unittest
import numpy as np
import ctypes
import os
import time

# Load Library
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libholo.so")

class RxState(ctypes.Structure):
    pass 

if os.path.exists(_lib_path):
    _lib = ctypes.CDLL(_lib_path)
    
    # helper to define args
    _lib.rx_create.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
    _lib.rx_create.restype = ctypes.POINTER(RxState)
    
    _lib.rx_destroy.argtypes = [ctypes.POINTER(RxState)]
    _lib.rx_destroy.restype = None
    
    _lib.rx_accumulate.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    _lib.rx_accumulate.restype = None
    
    _lib.rx_get_image.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
    _lib.rx_get_image.restype = None
    
    # Check if the new function exists (it won't yet)
    if hasattr(_lib, 'rx_solve_ls'):
        _lib.rx_solve_ls.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]
        _lib.rx_solve_ls.restype = None
        _lib.rx_get_ls_image.argtypes = [ctypes.POINTER(RxState), ctypes.POINTER(ctypes.c_uint8)]
        _lib.rx_get_ls_image.restype = None


class TestCUDALeastSquares(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(_lib_path):
            self.skipTest("libholo.so not found")
        
        self.width = 64
        self.height = 64
        self.npix = self.width * self.height
        self.seed = 12345
        
        # Create a synthetic image
        self.gt_image = np.zeros((self.height, self.width), dtype=np.float32)
        y, x = np.ogrid[:self.height, :self.width]
        self.gt_image = (np.sin(x/5) * np.cos(y/5) * 127 + 128).astype(np.float32)
        
        # We need to simulate the TRANSMITTER part to get measurements
        # Since we can't easily import the Tx part from this raw test without more boilerplate,
        # we'll reimplement a simple python version of the encoder just to generate 'b'
        # OR we can assume holo_protocol checks out.
        # Let's use the Python reference implementation from protocol.py if available, 
        # or just quick-and-dirty numpy here to ensure we feed the C decoder valid data.
        
        # Generate masks and measurements manually to be sure
        self.n_meas = 4096 # 100% sampling
        self.rng = np.random.default_rng(self.seed)
        
        # We need to match the C implementation's RNG exactly or consistent logic
        # attempting to match protocol.py's implementation:
        # It uses: rng.integers(0, 2) * 2 - 1
        # C uses: xoshiro256** or cuRAND? 
        # holo_cuda.cu uses cuRAND default. Achieving bit-exact match with numpy is hard.
        # BETTER STRATEGY: Use the library's OWN transmitter if possible.
        
        # Let's define TX bindings locally
        _lib.tx_create.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        _lib.tx_create.restype = ctypes.c_void_p
        _lib.tx_destroy.argtypes = [ctypes.c_void_p]
        _lib.tx_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        
        self.tx_state = _lib.tx_create(self.seed, self.width, self.height, 
                                       self.gt_image.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        self.measurements = np.zeros(self.n_meas, dtype=np.float32)
        _lib.tx_generate(self.tx_state, 
                         self.measurements.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
                         self.n_meas)
        
    def tearDown(self):
        if hasattr(self, 'tx_state') and self.tx_state:
            _lib.tx_destroy(self.tx_state)

    def test_ls_improvement(self):
        if not hasattr(_lib, 'rx_solve_ls'):
            print("Skipping LS test: rx_solve_ls not found in library")
            return

        rx = _lib.rx_create(self.seed, self.width, self.height)
        
        # 1. Matched Filter (Standard)
        _lib.rx_accumulate(rx, self.measurements.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.n_meas)
        
        mf_output = np.zeros(self.npix, dtype=np.uint8)
        _lib.rx_get_image(rx, mf_output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.n_meas)
        
        mf_psnr = self.calculate_psnr(self.gt_image.astype(np.uint8), mf_output.reshape(self.height, self.width))
        print(f"Matched Filter PSNR: {mf_psnr:.2f} dB")
        
        # 2. Least Squares
        # solve with 20 iterations
        start_t = time.time()
        _lib.rx_solve_ls(rx, self.measurements.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.n_meas, 50, 1e-6)
        ls_time = time.time() - start_t
        
        ls_output = np.zeros(self.npix, dtype=np.uint8)
        _lib.rx_get_ls_image(rx, ls_output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        
        ls_psnr = self.calculate_psnr(self.gt_image.astype(np.uint8), ls_output.reshape(self.height, self.width))
        print(f"Least Squares PSNR: {ls_psnr:.2f} dB (Time: {ls_time*1000:.1f} ms)")
        
        _lib.rx_destroy(rx)
        
        # Assert improvement
        # With 100% measurements, LS should be MUCH better
        self.assertGreater(ls_psnr, mf_psnr + 10, "LS should significantly outperform Matched Filter at 100% sampling")
        self.assertGreater(ls_psnr, 30.0, "LS should achieve high quality (>30dB)")

    def calculate_psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))

if __name__ == '__main__':
    unittest.main()
