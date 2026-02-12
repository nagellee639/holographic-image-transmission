import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr, LinearOperator
import time

def generate_mask(rng, num_pixels):
    return rng.integers(0, 2, size=num_pixels, dtype=np.int8) * 2 - 1

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# 1. Setup
SIZE = 32
NUM_PIXELS = SIZE * SIZE
measurements_counts = [100, 300, 500, 800, 1024, 1500, 2000]

# Create a dummy image (gradient + circle)
x = np.linspace(0, 1, SIZE)
y = np.linspace(0, 1, SIZE)
X, Y = np.meshgrid(x, y)
original_image = 127 + 127 * np.sin(10 * X) * np.cos(10 * Y)
original_image = (original_image + 50 * ((X-0.5)**2 + (Y-0.5)**2 < 0.1)).clip(0, 255).astype(np.float64)
image_flat = original_image.ravel()

print(f"Original Image Range: {original_image.min()} - {original_image.max()}")

results = {'count': [], 'matched': [], 'ls': [], 'l1': []}

# 2. Run Comparison
for num_meas in measurements_counts:
    print(f"\nTesting with {num_meas} measurements ({num_meas/NUM_PIXELS:.1%} sampling)...")
    
    # Generate Masks & Measurements
    # We use a fixed seed for reproducibility across counts if we wanted, 
    # but here we just regenerate for simplicity.
    rng = np.random.default_rng(42)
    
    # Matrix A: (M, N)
    # To use scipy solvers efficiently, we can construct the matrix explicitly for this small size
    # or use LinearOperator. For 32x32=1024 pixels, matrix is small.
    # explicit matrix generation
    A = np.zeros((num_meas, NUM_PIXELS), dtype=np.float32)
    b = np.zeros(num_meas, dtype=np.float32)
    
    for i in range(num_meas):
        mask = generate_mask(rng, NUM_PIXELS)
        A[i] = mask
        b[i] = np.dot(mask, image_flat)
        
    # --- Method 1: Matched Filter (Current) ---
    # x = A.T @ b / N
    # Note: Traditional Compressive Sensing matched filter is A.T @ y
    # But our protocol divides by N to normalize brightness directly.
    start = time.time()
    recon_matched = (A.T @ b) / num_meas
    # Normalize to 0-255 range as per current protocol
    mn, mx = recon_matched.min(), recon_matched.max()
    if mx - mn > 1e-6:
        recon_matched = (recon_matched - mn) / (mx - mn) * 255.0
    
    p_matched = psnr(original_image, recon_matched.reshape(SIZE, SIZE))
    t_matched = time.time() - start
    
    # --- Method 2: Least Squares (Iterative LSQR) ---
    # min ||Ax - b||^2
    start = time.time()
    res = lsqr(A, b, atol=1e-6, btol=1e-6)
    recon_ls = res[0]
    # No dynamic range normalization needed strictly if system is well-conditioned,
    # but clipping is good practice.
    recon_ls = recon_ls.clip(0, 255)
    
    p_ls = psnr(original_image, recon_ls.reshape(SIZE, SIZE))
    t_ls = time.time() - start
    
    print(f"  Matched Filter: PSNR = {p_matched:.2f} dB, Time = {t_matched*1000:.2f} ms")
    print(f"  Least Squares:  PSNR = {p_ls:.2f} dB, Time = {t_ls*1000:.2f} ms")
    
    results['count'].append(num_meas)
    results['matched'].append(p_matched)
    results['ls'].append(p_ls)

# 3. Plot
plt.figure(figsize=(10, 6))
plt.plot(results['count'], results['matched'], 'o-', label='Matched Filter (Current)')
plt.plot(results['count'], results['ls'], 's-', label='Least Squares (LSQR)')
plt.axvline(x=NUM_PIXELS, color='gray', linestyle='--', label='100% Sampling')
plt.xlabel('Number of Measurements')
plt.ylabel('PSNR (dB)')
plt.title(f'Decoding Quality vs Measurements ({SIZE}x{SIZE} Image)')
plt.grid(True)
plt.legend()
plt.savefig('comparison_plot.png')
print("\nPlot saved to comparison_plot.png")
