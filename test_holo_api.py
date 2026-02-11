
import unittest
import numpy as np
import ctypes
import os
from holo_api import HoloTx, HoloRx

class TestHoloAPI(unittest.TestCase):
    def setUp(self):
        # Create a dummy 64x48 image
        self.width = 64
        self.height = 48
        self.npix = self.width * self.height
        self.seed = 12345
        
        # Simple gradient image
        self.img = np.zeros((self.height, self.width), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                self.img[y, x] = (x + y) % 255

    def test_tx_lifecycle(self):
        """Test TX creation and destruction"""
        tx = HoloTx(self.seed, self.width, self.height, self.img)
        self.assertIsNotNone(tx.ptr)
        tx.close()
        self.assertIsNone(tx.ptr)

    def test_tx_generate(self):
        """Test generating samples"""
        tx = HoloTx(self.seed, self.width, self.height, self.img)
        count = 1000
        samples = tx.generate(count)
        
        self.assertEqual(len(samples), count)
        self.assertEqual(samples.dtype, np.float32)
        
        # Check that samples are not all zero (random mask * image != 0 usually)
        self.assertFalse(np.all(samples == 0))
        tx.close()

    def test_rx_lifecycle(self):
        """Test RX creation and destruction"""
        rx = HoloRx(self.seed, self.width, self.height)
        self.assertIsNotNone(rx.ptr)
        rx.close()
        self.assertIsNone(rx.ptr)

    def test_end_to_end_noiseless(self):
        """Test TX -> RX loopback matches original image (statistically)"""
        # 1. Setup TX and RX with same seed
        tx = HoloTx(self.seed, self.width, self.height, self.img)
        rx = HoloRx(self.seed, self.width, self.height)
        
        # 2. Generate many measurements (~2x pixels)
        n_meas = self.npix * 2
        chunk_size = 1024
        generated = 0
        
        while generated < n_meas:
            chunk = min(chunk_size, n_meas - generated)
            samples = tx.generate(chunk)
            rx.accumulate(samples)
            generated += chunk
            
        # 3. Reconstruct
        recon = rx.get_image()
        
        self.assertEqual(recon.shape, (self.height, self.width))
        self.assertEqual(recon.dtype, np.uint8)
        
        # 4. Check correlation / MSE
        # It won't be perfect, but should be close.
        # Mean Squared Error
        mse = np.mean((self.img.astype(float) - recon.astype(float))**2)
        psnr = 10 * np.log10(255**2 / mse)
        
        print(f"Mini-test PSNR: {psnr:.2f} dB")
        
        # We expect some reconstruction quality > 10dB for 2x measurements
        self.assertGreater(psnr, 10.0)
        
        tx.close()
        rx.close()

    def test_mismatched_dimensions(self):
        """Test error handling for wrong image size"""
        bad_img = np.zeros((10, 10), dtype=np.uint8) # 100 pixels
        with self.assertRaises(ValueError):
            HoloTx(self.seed, self.width, self.height, bad_img)

if __name__ == '__main__':
    unittest.main()
