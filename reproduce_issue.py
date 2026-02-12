
import numpy as np
from holo_api import HoloTx, GPU_AVAILABLE
import ctypes

def test_tx_uniqueness():
    if not GPU_AVAILABLE:
        print("Skipping GPU test (GPU not available)")
        return

    width, height = 64, 64
    img = np.zeros((height, width), dtype=np.uint8)
    # pattern
    img[20:40, 20:40] = 255
    
    print("Creating HoloTx...")
    tx = HoloTx(seed=12345, width=width, height=height, image_array=img)
    
    print("Generating batch 1 (Fast)...")
    m1 = tx.generate(100, fast=True)
    
    print("Generating batch 2 (Fast)...")
    m2 = tx.generate(100, fast=True)
    
    # Check if they are identical
    if np.allclose(m1, m2):
        print("[FAIL] Batch 1 and Batch 2 are IDENTICAL! Counter is likely resetting.")
    else:
        print("[PASS] Batch 1 and Batch 2 are different. Counter is working.")
        
    # Double check standard
    print("Generating batch 3 (Standard)...")
    m3 = tx.generate(100, fast=False)
    print("Generating batch 4 (Standard)...")
    m4 = tx.generate(100, fast=False)
    
    if np.allclose(m3, m4):
         print("[FAIL] Standard Batch 3 and 4 are IDENTICAL! (Unexpected for standard)")
    else:
         print("[PASS] Standard Batch 3 and 4 are different.")

    tx.close()

if __name__ == "__main__":
    try:
        test_tx_uniqueness()
    except Exception as e:
        print(f"An error occurred: {e}")
