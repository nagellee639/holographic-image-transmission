
import unittest
import sys
import threading
import time
from unittest.mock import MagicMock, patch

# Mock tkinter before importing gui
# We need to mock tk.Tk, ttk, filedialog, etc.
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.ImageTk'] = MagicMock()

# Mock sounddevice
sys.modules['sounddevice'] = MagicMock()

# Now import gui
from gui import HoloGUI
import numpy as np

class TestGUIHeadless(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()
        self.app = HoloGUI(self.root)
        
        # Manually inject dummy image since we can't load from file
        self.app.tx_img = np.zeros((480, 640), dtype=np.uint8)
        
        # Stop actual threads from spinning forever
        self.app.is_transmitting = False
        self.app.is_playing = False

    def test_tx_start_stop(self):
        """Test transmission toggling logic"""
        # Mock load_image result
        with patch('gui.HoloTx') as MockTx:
             # Simulate image loaded
             self.app.tx = MockTx.return_value
             
             # Start
             self.app.toggle_tx()
             self.assertTrue(self.app.is_transmitting)
             self.assertIn("Stop", str(self.app.btn_tx.config.call_args))
             
             # Stop
             self.app.toggle_tx()
             self.assertFalse(self.app.is_transmitting)
             self.assertIn("Start", str(self.app.btn_tx.config.call_args))

    def test_audio_buffering(self):
        """Test audio buffer accumulation logic"""
        # Simulate TX thread generating data
        chunk = np.random.rand(1024).astype(np.float32)
        
        with self.app.lock:
            self.app.audio_chunks.append(chunk)
            self.app.valid_samples += len(chunk)
            
        self.assertEqual(self.app.valid_samples, 1024)
        self.assertEqual(len(self.app.audio_chunks), 1)

    def test_play_logic_no_device(self):
        """Test graceful degradation when sounddevice fails"""
        # Mock OutputStream raising Exception
        sys.modules['sounddevice'].OutputStream.side_effect = Exception("No Audio Device")
        
        # Setup data
        self.app.valid_samples = 2000
        
        # Start Play
        self.app.toggle_play()
        
        # Should catch exception, show error (mocked), and start pseudo-thread
        # Verify state is playing (silent mode)
        self.assertTrue(self.app.is_playing)
        
        # Stop Play
        self.app.toggle_play()
        self.assertFalse(self.app.is_playing)

    def test_audio_unavailable(self):
        """Test fallback when sounddevice library is completely missing"""
        # Patch the global AUDIO_AVAILABLE flag in gui module
        with patch('gui.AUDIO_AVAILABLE', False):
            # Must have data to play
            self.app.valid_samples = 100
            
            # Start Play
            self.app.toggle_play()
            
            # Should enter pseudo-play mode directly
            self.assertTrue(self.app.is_playing)
            
            # Verify no attempt to create OutputStream
            self.assertEqual(sys.modules['sounddevice'].OutputStream.call_count, 0)
            
            # Stop
            self.app.toggle_play()

if __name__ == '__main__':
    unittest.main()
