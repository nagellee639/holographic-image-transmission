import numpy as np
import sys
import os
try:
    import holo_api
    from holo_api import HoloTx, HoloRx
except ImportError:
    sys.path.append(os.getcwd())
    import holo_api
    from holo_api import HoloTx, HoloRx

def test_fista_bands():
    if not holo_api.GPU_AVAILABLE:
        print("[SKIP] GPU not available.")
        return

    width, height = 64, 64
    seed = 42

    # Create a flat gray image - simplest test for bands
    # If bands appear, they will be obvious deviations from 128
    img = np.full((height, width), 128, dtype=np.uint8)

    print("Creating HoloTx...")
    tx = HoloTx(seed=seed, width=width, height=height, image_array=img)

    # Generate enough measurements for full reconstruction
    # 64*64 = 4096 pixels. 
    # Use 2x oversampling (8192) to be safe for L1
    n_meas = 8192
    print(f"Generating {n_meas} measurements...")
    measurements, _ = tx.generate(n_meas, fast=True)
    tx.close()

    print("Creating HoloRx...")
    rx = HoloRx(seed=seed, width=width, height=height)
    
    # Create Solver (L1, Fast) - Type 20
    print("Creating Solver (L1 Fast)...")
    solver = rx.create_solver_l1(measurements, lambda_val=0.1, fast=True)

    # Step solver
    n_iters = 50
    print(f"Running {n_iters} iterations...")
    for i in range(n_iters):
        solver.step()
        if i % 10 == 0:
            rec = solver.get_image()
            mse = np.mean((rec.astype(float) - img.astype(float))**2)
            print(f"Iter {i}: MSE = {mse:.2f}")

    rec = solver.get_image()
    solver.close()
    rx.close()

    # Analyze for bands
    # Check row variance vs column variance
    # If horizontal bands, row variance (variance of row means) will be high
    # If vertical bands, col variance (variance of col means) will be high
    
    row_means = np.mean(rec, axis=1)
    col_means = np.mean(rec, axis=0)
    
    row_var = np.var(row_means)
    col_var = np.var(col_means)
    
    # Ideally should be close to 0 for flat image
    print(f"Row Means Variance: {row_var:.4f}")
    print(f"Col Means Variance: {col_var:.4f}")
    
    mse = np.mean((rec.astype(float) - img.astype(float))**2)
    print(f"Final MSE: {mse:.2f}")

    if mse > 100: # Arbitrary high threshold for "bad reconstruction"
        print("[FAIL] MSE is high, reconstruction failed.")
    elif row_var > 10 or col_var > 10:
        print(f"[FAIL] High variance in row/col means detected! Potential banding. RowVar={row_var:.2f}, ColVar={col_var:.2f}")
    else:
        print("[PASS] Reconstruction looks flat.")

if __name__ == "__main__":
    test_fista_bands()
