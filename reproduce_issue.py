
import numpy as np
import sys
import os
import ctypes

# Import the module
import alt_protocols

def check_protocol(name, TxCls, RxCls):
    print(f"\nChecking {name}...")
    W, H = 64, 64
    seed = 42
    
    # Create a simple pattern: Left half 0, Right half 255
    img = np.zeros((H, W), dtype=np.float32)
    img[:, 32:] = 255.0
    
    tx = TxCls(seed, W, H, img)
    rx = RxCls(seed, W, H)
    
    # Generate measurements
    count = 10000
    samples = tx.generate(count)
    rx.accumulate(samples)
    
    recon = rx.get_image()
    
    # Check stats
    print(f"  Measurements: {len(samples)}")
    print(f"  Recon stats: min={recon.min()}, max={recon.max()}, mean={recon.mean()}")
    
    # Check center pixels
    mid_left = recon[32, 16]
    mid_right = recon[32, 48]
    print(f"  Left pixel (should be dark): {mid_left}")
    print(f"  Right pixel (should be bright): {mid_right}")
    
    if recon.min() == 128 and recon.max() == 128:
        print("  [FAIL] Image is completely gray (128)!")
    elif mid_left > 50 or mid_right < 200:
        print("  [WARN] Image contract is low.")
    else:
        print("  [PASS] Image looks plausible.")

    # Check internal counts if possible
    zero_counts = np.sum(rx.counts == 0)
    print(f"  Pixels with 0 updates: {zero_counts}/{W*H} ({zero_counts/(W*H)*100:.1f}%)")

def main():
    print(f"Lib loaded: {alt_protocols._lib is not None}")
    
    check_protocol("Random Line", alt_protocols.LineTx, alt_protocols.LineRx)
    check_protocol("Random Block", alt_protocols.BlockTx, alt_protocols.BlockRx)

if __name__ == "__main__":
    main()
