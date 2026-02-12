import time
import numpy as np
import holo_api
import argparse

def benchmark(width, height, n_meas, iterations):
    print(f"Benchmarking: {width}x{height} pixels, {n_meas} samples, {iterations} iterations")
    
    seed = 42
    # Create fake image
    tx_img = np.zeros((height, width), dtype=np.uint8)
    # Draw a rect
    tx_img[height//4:3*height//4, width//4:3*width//4] = 200
    
    print("Generating measurements...")
    t0 = time.time()
    tx = holo_api.HoloTx(seed, width, height, tx_img)
    meas = tx.generate(n_meas)
    tx.close()
    print(f"Generation took {time.time()-t0:.2f}s")
    
    rx = holo_api.HoloRx(seed, width, height)
    
    print("\n--- L2 Solver (CGLS) ---")
    s_ls = rx.create_solver_ls(meas)
    
    # Warmup
    # s_ls.step() 
    
    t_start = time.time()
    for i in range(iterations):
        t_iter = time.time()
        s_ls.step()
        print(f"Iter {i+1}: {time.time() - t_iter:.4f}s")
        
    total_time = time.time() - t_start
    print(f"L2 Average time per iteration: {total_time/iterations:.4f}s")
    s_ls.close()
    
    # Skip L1 for massive sizes if L2 is already slow, or run fewer iters
    print("\n--- L1 Solver (ISTA) ---")
    s_l1 = rx.create_solver_l1(meas, lambda_val=0.5)
    
    t_start = time.time()
    for i in range(iterations):
        t_iter = time.time()
        s_l1.step()
        print(f"Iter {i+1}: {time.time() - t_iter:.4f}s")
        
    total_time = time.time() - t_start
    print(f"L1 Average time per iteration: {total_time/iterations:.4f}s")
    s_l1.close()
    
    rx.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=300000)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    
    # Check if GPU available
    if not holo_api.GPU_AVAILABLE:
        print("GPU not available, cannot benchmark CUDA solvers.")
        exit(1)
        
    benchmark(args.width, args.height, args.samples, args.iters)
