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

def estimate_spectral_norm(rx, n_meas, n_iters=20):
    # Power iteration to estimate largest eigenvalue of A^T A
    # A^T A x = ?
    # We can use the solver's internal operators if we had access, 
    # but we can simulate it with generate/accumulate if we are careful.
    # Actually, HoloTx generates Ax. HoloRx accumulates A^T y.
    # So A^T A x:
    # 1. b = Ax (using HoloTx with image x)
    # 2. x_new = A^T b (using HoloRx accumulate)
    
    width = rx.width
    height = rx.height
    
    x = np.random.randn(height, width).astype(np.float32)
    x /= np.linalg.norm(x)
    
    print(f"Estimating spectral norm (Power Iteration, {n_iters} iters)...")
    
    for i in range(n_iters):
        # 1. Ax
        # We need a Tx consistent with Rx
        # Note: Tx generates new masks every time unless we control it.
        # But 'fast' generation uses consistent indexing if we reset.
        # Actually, creating a new Tx each time is slow but safe for "stateless" A.
        # But we need A to be CONSTANT.
        # The 'fast' mode relies on (seed, idx).
        # So we can effectively simulate specific A.
        
        # ACTUALLY, checking current implementation:
        # The solver uses internal A/A^T.
        # It's hard to extract them without modifying C++.
        # Let's just create a solver and check its behavior with a smaller step size.
        pass
        
    print("Cannot easily do power iteration without C++ access. Skipping.")

def test_stability_with_step_size():
    if not holo_api.GPU_AVAILABLE:
        return

    width, height = 64, 64
    seed = 42
    img = np.full((height, width), 128, dtype=np.uint8)

    # Generate meas
    tx = HoloTx(seed=seed, width=width, height=height, image_array=img)
    n_meas = 8192 # 2x oversampling
    measurements, _ = tx.generate(n_meas, fast=True)
    tx.close()

    rx = HoloRx(seed=seed, width=width, height=height)
    
    # We can't change step size from Python easily without modifying C++ or hacking
    # But we can try the solver step-by-step and see divergence.
    
    # Let's rely on the C++ fix.
    # But we can confirm the Divergence is robustly reproducible.
    
    print("Reproducing Divergence...")
    solver = rx.create_solver_l1(measurements, lambda_val=0.1, fast=True)
    
    diverged = False
    prev_mse = float('inf')
    
    for i in range(100):
        solver.step()
        if i % 5 == 0:
            rec = solver.get_image()
            mse = np.mean((rec.astype(float) - img.astype(float))**2)
            print(f"Iter {i}: MSE={mse:.2f}")
            if mse > 1e5:
                print("Diverged massively!")
                diverged = True
                break
            if i > 10 and mse > prev_mse * 1.5:
                print("MSE increasing significantly!")
                diverged = True
                break
            if i > 10:
                prev_mse = mse
                
    if diverged:
        print("[FAIL] Solver Diverged. Step size likely too high.")
    else:
        print("[PASS] Solver Stable.")

    solver.close()
    rx.close()

if __name__ == "__main__":
    test_stability_with_step_size()
