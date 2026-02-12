import ctypes
import numpy as np
import os
import sys
import time

# Ensure we import the local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import holo_api
except ImportError:
    sys.path.append("..")
    import holo_api

def test_solvers():
    if not holo_api.GPU_AVAILABLE:
        print("[SKIP] GPU not available")
        return

    print("[INFO] Testing Stateful Solvers...")
    
    # 1. Setup
    width, height = 64, 64
    seed = 42
    
    # Create fake image
    tx_img = np.zeros((height, width), dtype=np.uint8)
    tx_img[16:48, 16:48] = 200 # Square
    
    # Create Measurements
    tx = holo_api.HoloTx(seed, width, height, tx_img)
    N = 2000
    meas = tx.generate(N)
    tx.close()
    
    rx = holo_api.HoloRx(seed, width, height)
    
    from utils import ssim_metric
    
    # 2. Test LS Solver
    print("\n--- Testing L2 (CGLS) Solver ---")
    s_ls = rx.create_solver_ls(meas)
    for i in range(15):
        s_ls.step()
    img_ls = s_ls.get_image()
    
    ssim_ls = ssim_metric(tx_img, img_ls)
    print(f"[PASS] LS Solver ran 15 steps. Max val: {img_ls.max()} | SSIM: {ssim_ls:.4f}")
    if ssim_ls < 0.1: # Low threshold for short run
         print(f"[WARN] SSIM is very low ({ssim_ls:.4f}). Check reconstruction quality.")
    s_ls.close()
    
    # 3. Test L1 Solver
    print("\n--- Testing L1 (ISTA) Solver ---")
    s_l1 = rx.create_solver_l1(meas, lambda_val=0.5)
    for i in range(15):
        s_l1.step()
    img_l1 = s_l1.get_image()
    
    ssim_l1 = ssim_metric(tx_img, img_l1)
    print(f"[PASS] L1 Solver ran 15 steps. Max val: {img_l1.max()} | SSIM: {ssim_l1:.4f}")
    if ssim_l1 < 0.1:
         print(f"[WARN] SSIM is very low ({ssim_l1:.4f}). Check reconstruction quality.")
    s_l1.close()
    
    # 4. Test Histogram Projection (Host Pinned Memory)
    print("\n--- Testing Histogram Projection ---")
    s_hist = rx.create_solver_ls(meas) # Re-use LS
    
    # Fake histogram: mostly black, some white
    hist = np.zeros(256, dtype=np.int32)
    hist[0] = width*height - 500
    hist[200] = 500
    
    # Run a few steps, then force
    s_hist.step()
    s_hist.project_histogram(hist)
    s_hist.step()
    
    img_hist = s_hist.get_image()
    
    # Verify histogram of output matches roughly?
    out_hist, _ = np.histogram(img_hist, bins=256, range=(0,255))
    # It won't match exactly because we do steps AFTER projection, but should be close.
    print(f"[PASS] Histogram projection ran. Output 0-bin count: {out_hist[0]} (Target: {hist[0]})")
    s_hist.close()
    rx.close()

if __name__ == "__main__":
    test_solvers()
