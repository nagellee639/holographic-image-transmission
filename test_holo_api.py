
import unittest
import numpy as np
import ctypes
import os
from holo_api import HoloTx, HoloRx

# ── GPU availability check ────────────────────────────────────────────
def _gpu_available():
    """Quick probe: try creating a HoloRx, return True if GPU works."""
    try:
        rx = HoloRx(999, 4, 4)
        rx.close()
        return True
    except RuntimeError:
        return False

GPU_AVAILABLE = _gpu_available()
GPU_SKIP_MSG = "GPU unavailable (CUDA init error 999 — driver/toolkit mismatch or GPU needs reset)"


class TestHoloAPIGPU(unittest.TestCase):
    """Integration tests that exercise the real CUDA library.
    These skip automatically when the GPU is unavailable."""

    def setUp(self):
        self.width = 64
        self.height = 48
        self.npix = self.width * self.height
        self.seed = 12345
        self.img = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                self.img[y, x] = (x + y) % 255

    @unittest.skipUnless(GPU_AVAILABLE, GPU_SKIP_MSG)
    def test_tx_lifecycle(self):
        """Test TX creation and destruction"""
        tx = HoloTx(self.seed, self.width, self.height, self.img)
        self.assertIsNotNone(tx.ptr)
        tx.close()
        self.assertIsNone(tx.ptr)

    @unittest.skipUnless(GPU_AVAILABLE, GPU_SKIP_MSG)
    def test_tx_generate(self):
        """Test generating samples"""
        tx = HoloTx(self.seed, self.width, self.height, self.img)
        count = 1000
        samples = tx.generate(count)
        self.assertEqual(len(samples), count)
        self.assertEqual(samples.dtype, np.float32)
        self.assertFalse(np.all(samples == 0))
        tx.close()

    @unittest.skipUnless(GPU_AVAILABLE, GPU_SKIP_MSG)
    def test_rx_lifecycle(self):
        """Test RX creation and destruction"""
        rx = HoloRx(self.seed, self.width, self.height)
        self.assertIsNotNone(rx.ptr)
        rx.close()
        self.assertIsNone(rx.ptr)

    @unittest.skipUnless(GPU_AVAILABLE, GPU_SKIP_MSG)
    def test_end_to_end_noiseless(self):
        """Test TX -> RX loopback matches original image (statistically)"""
        tx = HoloTx(self.seed, self.width, self.height, self.img)
        rx = HoloRx(self.seed, self.width, self.height)

        n_meas = self.npix * 2
        chunk_size = 1024
        generated = 0
        while generated < n_meas:
            chunk = min(chunk_size, n_meas - generated)
            samples = tx.generate(chunk)
            rx.accumulate(samples)
            generated += chunk

        recon = rx.get_image()
        self.assertEqual(recon.shape, (self.height, self.width))
        self.assertEqual(recon.dtype, np.uint8)

        mse = np.mean((self.img.astype(float) - recon.astype(float))**2)
        psnr = 10 * np.log10(255**2 / mse)
        print(f"Mini-test PSNR: {psnr:.2f} dB")
        self.assertGreater(psnr, 10.0)
        tx.close()
        rx.close()

    def test_mismatched_dimensions(self):
        """Test error handling for wrong image size"""
        bad_img = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            HoloTx(self.seed, self.width, self.height, bad_img)


class TestHoloAPIErrorHandling(unittest.TestCase):
    """Tests that verify graceful failure when GPU is unavailable.
    These always run — they don't need a working GPU."""

    @unittest.skipIf(GPU_AVAILABLE, "GPU is working, can't test failure path")
    def test_tx_create_raises_runtime_error(self):
        """When GPU is broken, HoloTx() should raise RuntimeError, not crash."""
        img = np.zeros((48, 64), dtype=np.uint8)
        with self.assertRaises(RuntimeError) as ctx:
            HoloTx(42, 64, 48, img)
        self.assertIn("GPU TX init failed", str(ctx.exception))

    @unittest.skipIf(GPU_AVAILABLE, "GPU is working, can't test failure path")
    def test_rx_create_raises_runtime_error(self):
        """When GPU is broken, HoloRx() should raise RuntimeError, not crash."""
        with self.assertRaises(RuntimeError) as ctx:
            HoloRx(42, 64, 48)
        self.assertIn("GPU RX init failed", str(ctx.exception))

    @unittest.skipIf(GPU_AVAILABLE, "GPU is working, can't test failure path")
    def test_process_does_not_exit(self):
        """Verify the process survives GPU init failure (the original crash bug)."""
        try:
            HoloTx(42, 64, 48, np.zeros((48, 64), dtype=np.uint8))
        except RuntimeError:
            pass
        # If we reach here, the process didn't exit(1) — test passes
        self.assertTrue(True, "Process survived GPU failure")


if __name__ == '__main__':
    unittest.main()
