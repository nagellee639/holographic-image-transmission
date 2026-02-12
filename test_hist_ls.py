import unittest
import numpy as np
import ctypes
from holo_api import HoloRx, RxState, _lib

class TestHistogramLS(unittest.TestCase):
    def test_histogram_projection(self):
        """Test that the histogram solver enforces the target distribution."""
        W, H = 64, 64
        rx = HoloRx(W, H, seed=12345)
        
        # Create a synthetic image with a specific histogram
        # Half black (0), half white (255)
        img = np.zeros(W*H, dtype=np.uint8)
        img[W*H//2:] = 255
        
        # Exact histogram
        hist_orig, _ = np.histogram(img, bins=256, range=(0, 255))
        hist_orig = hist_orig.astype(np.int32)
        
        # Fake measurements (noise)
        # We don't need real measurements to test the projection constraint.
        # The projection happens at the end of the loop.
        # Even if CGLS fails to find a good solution for 'b', the final 'x' MUST obey the histogram.
        
        # Let's generate dummy measurements
        rx.measurements_count = 100
        meas = np.random.randn(100).astype(np.float32)
        
        # Solve
        # outer 1, inner 1 to be fast. 
        # The projection happens at the end of outer iter.
        res_img = rx.solve_histogram_ls(meas, hist_orig, outer_iters=2, inner_iters=1)
        
        # Verify 1: Output size
        self.assertEqual(res_img.shape, (H, W))
        
        # Verify 2: Histogram match
        # The output pixels are uint8. 
        # The distribution should identical to hist_orig.
        
        hist_res, _ = np.histogram(res_img.flatten(), bins=256, range=(0, 255))
        
        # Compare histograms
        # Note: rounding errors during float->uint8 conversion might cause slight bin hopping (off by 1).
        # But global stats should match.
        
        diff = np.abs(hist_orig - hist_res).sum()
        print(f"Histogram Diff: {diff} pixels out of {W*H}")
        
        # Allow small deviation due to rounding
        self.assertLess(diff, 50, "Output histogram should closely match target")

if __name__ == '__main__':
    unittest.main()
