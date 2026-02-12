import ctypes
import numpy as np
import os
import sys

# Ensure we import the local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import holo_api
except ImportError:
    # If run directly from tests/
    sys.path.append("..")
    import holo_api

def test_holo_cuda_rx_histogram_ls():
    if not holo_api.GPU_AVAILABLE:
        print("[SKIP] GPU not available, skipping CUDA test")
        return

    print("[INFO] GPU available, testing rx_solve_histogram_ls symbol...")
    
    # 1. Instantiate HoloRx
    width, height = 64, 64
    seed = 12345
    rx = holo_api.HoloRx(seed, width, height)
    
    # 2. Prepare dummy data
    N = 100
    measurements = np.random.randn(N).astype(np.float32)
    # create a dummy histogram (must sum to npix)
    # simplistic: just put all pixels in one bin for testing
    histogram = np.zeros(256, dtype=np.int32)
    histogram[0] = width * height
    
    # 3. Call the function that was missing
    try:
        rx.solve_histogram_ls(measurements, histogram, outer_iters=1, inner_iters=1)
        print("[PASS] rx_solve_histogram_ls called successfully")
    except AttributeError:
        print("[FAIL] rx_solve_histogram_ls symbol missing in libholo.so")
        sys.exit(1)
    except Exception as e:
        print(f"[FAIL] verify exception: {e}")
        sys.exit(1)
    finally:
        rx.close()

if __name__ == "__main__":
    test_holo_cuda_rx_histogram_ls()
