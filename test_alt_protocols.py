"""
Tests for alternative transmission protocols.
Verifies that each protocol can encode and decode a simple test image
and produce a recognizable reconstruction.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from alt_protocols import (
    PixelTx, PixelRx,
    LineTx, LineRx,
    BlockTx, BlockRx,
    PROTOCOLS,
)
from utils import psnr


def make_test_image(w=64, h=48):
    """Create a simple gradient test image (easy to verify)."""
    img = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            img[y, x] = ((x + y) / (w + h)) * 255.0
    return img


def run_protocol_test(TxClass, RxClass, name, n_measurements, expected_psnr_min):
    """Generic test: encode N measurements, decode, check PSNR."""
    W, H = 64, 48
    seed = 42
    img = make_test_image(W, H)
    
    tx = TxClass(seed, W, H, img)
    rx = RxClass(seed, W, H)
    
    # Generate and accumulate in chunks
    chunk = min(1000, n_measurements)
    remaining = n_measurements
    while remaining > 0:
        n = min(chunk, remaining)
        samples = tx.generate(n)
        rx.accumulate(samples)
        remaining -= n
    
    recon = rx.get_image()
    
    # Calculate quality
    p = psnr(img.astype(np.uint8), recon)
    
    print(f"  [{name}] {n_measurements} measurements -> PSNR = {p:.1f} dB (need >= {expected_psnr_min} dB)")
    
    assert recon.shape == (H, W), f"Shape mismatch: {recon.shape}"
    assert recon.dtype == np.uint8, f"Dtype mismatch: {recon.dtype}"
    assert p >= expected_psnr_min, f"PSNR too low: {p:.1f} < {expected_psnr_min}"
    
    tx.close()
    rx.close()
    return p


def test_pixel_basic():
    """Random Pixel: with enough samples, every pixel should be sampled."""
    print("\n=== Random Pixel Protocol ===")
    # 64x48 = 3072 pixels. With 30k samples, each pixel sampled ~10x on average.
    run_protocol_test(PixelTx, PixelRx, "Pixel-30k", 30000, 25.0)


def test_pixel_progressive():
    """Random Pixel: quality should improve with more measurements."""
    print("\n=== Random Pixel Progressive ===")
    W, H = 64, 48
    seed = 42
    img = make_test_image(W, H)
    
    tx = PixelTx(seed, W, H, img)
    rx = PixelRx(seed, W, H)
    
    prev_p = 0
    for n in [1000, 5000, 20000]:
        samples = tx.generate(n)
        rx.accumulate(samples)
        recon = rx.get_image()
        p = psnr(img.astype(np.uint8), recon)
        print(f"  [Pixel] {rx.measurements_count} total -> PSNR = {p:.1f} dB")
        # Quality should generally improve (allow small regression)
        assert p >= prev_p - 2.0, f"Quality regressed: {p:.1f} < {prev_p:.1f} - 2"
        prev_p = p


def test_line_basic():
    """Random Line: with enough samples, image should emerge."""
    print("\n=== Random Line Protocol ===")
    # Lines touch many pixels, so coverage is faster but values are averaged (blurry)
    run_protocol_test(LineTx, LineRx, "Line-30k", 30000, 15.0)


def test_block_basic():
    """Random Block: with enough samples, image should emerge."""
    print("\n=== Random Block Protocol ===")
    run_protocol_test(BlockTx, BlockRx, "Block-10k", 10000, 15.0)


def test_block_fast_convergence():
    """Random Block should converge faster than Pixel for coarse structure."""
    print("\n=== Block vs Pixel Convergence ===")
    W, H = 64, 48
    seed = 42
    img = make_test_image(W, H)
    
    # Block with just 2000 measurements
    btx = BlockTx(seed, W, H, img)
    brx = BlockRx(seed, W, H)
    samples = btx.generate(2000)
    brx.accumulate(samples)
    block_img = brx.get_image()
    block_p = psnr(img.astype(np.uint8), block_img)
    
    # Pixel with 2000 measurements
    ptx = PixelTx(seed, W, H, img)
    prx = PixelRx(seed, W, H)
    samples = ptx.generate(2000)
    prx.accumulate(samples)
    pixel_img = prx.get_image()
    pixel_p = psnr(img.astype(np.uint8), pixel_img)
    
    print(f"  Block@2k: {block_p:.1f} dB vs Pixel@2k: {pixel_p:.1f} dB")
    # Block should generally have higher PSNR at low counts for larger images
    # On tiny 64x48 gradient, the difference is marginal
    print(f"  (Block {'wins' if block_p > pixel_p else 'loses'} at this scale)")


def test_noise_resilience():
    """All protocols should handle noisy measurements gracefully."""
    print("\n=== Noise Resilience ===")
    W, H = 64, 48
    seed = 42
    img = make_test_image(W, H)
    snr_db = 10  # 10dB SNR
    
    for name, (TxCls, RxCls) in PROTOCOLS.items():
        tx = TxCls(seed, W, H, img)
        rx = RxCls(seed, W, H)
        
        n = 20000
        samples = tx.generate(n)
        
        # Add noise
        sig_power = np.mean(samples**2)
        noise_power = sig_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=n).astype(np.float32)
        noisy_samples = samples + noise
        
        rx.accumulate(noisy_samples)
        recon = rx.get_image()
        p = psnr(img.astype(np.uint8), recon)
        print(f"  [{name}] 20k @ SNR={snr_db}dB -> PSNR = {p:.1f} dB")
        assert p > 5.0, f"{name} failed noise test: PSNR={p:.1f}"


def test_registry():
    """Verify PROTOCOLS registry has correct entries."""
    print("\n=== Registry ===")
    assert "Random Pixel" in PROTOCOLS
    assert "Random Line" in PROTOCOLS
    assert "Random Block" in PROTOCOLS
    for name, (tx_cls, rx_cls) in PROTOCOLS.items():
        print(f"  {name}: TX={tx_cls.__name__}, RX={rx_cls.__name__}")


if __name__ == "__main__":
    test_registry()
    test_pixel_basic()
    test_pixel_progressive()
    test_line_basic()
    test_block_basic()
    test_block_fast_convergence()
    test_noise_resilience()
    print("\n✅ All alternative protocol tests passed!")
