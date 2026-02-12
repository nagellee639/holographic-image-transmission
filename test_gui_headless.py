"""
Headless tests for every GUI function in HoloGUI.

Mocks tkinter, PIL.ImageTk, and sounddevice so tests run without a display.
Each test targets a specific button / function that the GUI exposes.

Run:  python3 -m pytest test_gui_headless.py -v
"""

import unittest
import sys
import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock, call

# ── Mock GUI toolkit imports before importing gui ──────────────────────
mock_tk = MagicMock()
mock_ttk = MagicMock()
mock_filedialog = MagicMock()
mock_messagebox = MagicMock()
mock_pil = MagicMock()
mock_pil_image = MagicMock()
mock_pil_imagetk = MagicMock()
mock_sd = MagicMock()

# CRITICAL: Make StringVar and BooleanVar return unique instances per call.
# Without this, all var_proto/var_mode/var_file/etc. become the same object.
mock_tk.StringVar = lambda **kw: MagicMock(**kw)
mock_tk.BooleanVar = lambda **kw: MagicMock(**kw)

sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.ttk'] = mock_ttk
sys.modules['tkinter.filedialog'] = mock_filedialog
sys.modules['tkinter.messagebox'] = mock_messagebox
sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_pil_image
sys.modules['PIL.ImageTk'] = mock_pil_imagetk
sys.modules['sounddevice'] = mock_sd

# Now import gui
import gui
from gui import HoloGUI, RadioChannel
import numpy as np


class _GUITestBase(unittest.TestCase):
    """Base class that creates a HoloGUI with mocked tkinter."""

    def setUp(self):
        # Patch HoloTx / HoloRx so the C library is never loaded during tests
        self._patcher_tx = patch.object(gui, 'HoloTx')
        self._patcher_rx = patch.object(gui, 'HoloRx')
        self.MockTx = self._patcher_tx.start()
        self.MockRx = self._patcher_rx.start()

        # Give the mock RX a measurements_count attribute
        self.MockRx.return_value.measurements_count = 0

        self.root = MagicMock()
        self.app = HoloGUI(self.root)

        # Wire up StringVar mocks so .get() returns sensible defaults
        self.app.var_proto.get = MagicMock(return_value="Dense Holographic")
        self.app.var_mode.get = MagicMock(return_value="FM (12 kHz)")
        self.app.var_snr.get = MagicMock(return_value="Perfect")
        self.app.var_qsb.get = MagicMock(return_value=False)
        self.app.var_file.get = MagicMock(return_value="transmission.wav")
        self.app.var_live.get = MagicMock(return_value=True)

        # Inject a dummy image so image-dependent functions work
        self.app.tx_img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

        # Prevent real threads from spinning
        self.app.is_transmitting = False
        self.app.is_playing = False

        # Reset the messagebox mock used inside gui module
        gui.messagebox = MagicMock()
        # Make filedialog accessible for tests
        gui.filedialog = MagicMock()

    def tearDown(self):
        self._patcher_tx.stop()
        self._patcher_rx.stop()

    def _populate_audio(self, n_samples=4096):
        """Helper: fill audio_chunks with real numpy data."""
        chunk = np.random.rand(n_samples).astype(np.float32)
        with self.app.lock:
            self.app.audio_chunks = [chunk]
            self.app.valid_samples = n_samples


# ======================================================================
# 1. load_image
# ======================================================================
class TestLoadImage(_GUITestBase):

    def test_load_image_cancel_dialog(self):
        """Cancelling the file dialog should do nothing."""
        gui.filedialog.askopenfilename.return_value = ""
        old_tx = self.app.tx
        self.app.load_image()
        self.assertEqual(self.app.tx, old_tx)

    def test_load_image_valid_file(self):
        """Loading a valid .png should set tx_img and create a TX."""
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        try:
            with patch.object(gui, 'Image') as MockImage, \
                 patch.object(gui, 'ImageTk'):
                mock_pil_obj = MagicMock()
                mock_pil_obj.convert.return_value = mock_pil_obj
                mock_pil_obj.resize.return_value = mock_pil_obj
                MockImage.open.return_value = mock_pil_obj
                MockImage.LANCZOS = 1

                with patch.object(gui.np, 'asarray', return_value=np.zeros((480, 640), dtype=np.uint8)):
                    self.app.load_image(tmp.name)

                MockImage.open.assert_called_once_with(tmp.name)
                self.MockTx.assert_called()
        finally:
            os.unlink(tmp.name)

    def test_load_image_missing_file(self):
        """Loading a nonexistent file should show an error, not crash."""
        self.app.load_image("/nonexistent/path/image.png")
        gui.messagebox.showerror.assert_called_once()
        self.assertIn("not found", str(gui.messagebox.showerror.call_args))

    def test_load_image_non_image_file(self):
        """Loading a .wav file should show unsupported format error."""
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        try:
            self.app.load_image(tmp.name)
            gui.messagebox.showerror.assert_called_once()
            self.assertIn("Unsupported", str(gui.messagebox.showerror.call_args))
        finally:
            os.unlink(tmp.name)

    def test_load_image_mp3_rejected(self):
        """Loading a .mp3 should be rejected with a clear error."""
        tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmp.close()
        try:
            self.app.load_image(tmp.name)
            gui.messagebox.showerror.assert_called_once()
            self.assertIn("Unsupported", str(gui.messagebox.showerror.call_args))
        finally:
            os.unlink(tmp.name)


# ======================================================================
# 2. set_mode
# ======================================================================
class TestSetMode(_GUITestBase):

    def test_set_mode_cw(self):
        self.app.var_mode.get.return_value = "CW (100 Hz)"
        self.app.set_mode()
        self.assertEqual(self.app.sample_rate, 100)

    def test_set_mode_ssb(self):
        self.app.var_mode.get.return_value = "SSB (3 kHz)"
        self.app.set_mode()
        self.assertEqual(self.app.sample_rate, 3000)

    def test_set_mode_fm(self):
        self.app.var_mode.get.return_value = "FM (12 kHz)"
        self.app.set_mode()
        self.assertEqual(self.app.sample_rate, 12000)

    def test_set_mode_updates_channel(self):
        """Changing mode should also update channel.sr."""
        self.app.var_mode.get.return_value = "CW (100 Hz)"
        self.app.set_mode()
        self.assertEqual(self.app.channel.sr, 100)


# ======================================================================
# 3. update_channel
# ======================================================================
class TestUpdateChannel(_GUITestBase):

    def test_snr_perfect(self):
        self.app.var_snr.get.return_value = "Perfect"
        self.app.update_channel()
        self.assertIsNone(self.app.channel.snr_db)

    def test_snr_30db(self):
        self.app.var_snr.get.return_value = "Strong (30dB)"
        self.app.update_channel()
        self.assertEqual(self.app.channel.snr_db, 30)

    def test_snr_20db(self):
        self.app.var_snr.get.return_value = "Moderate (20dB)"
        self.app.update_channel()
        self.assertEqual(self.app.channel.snr_db, 20)

    def test_snr_10db(self):
        self.app.var_snr.get.return_value = "Weak (10dB)"
        self.app.update_channel()
        self.assertEqual(self.app.channel.snr_db, 10)

    def test_snr_0db(self):
        self.app.var_snr.get.return_value = "Deep Fade (0dB)"
        self.app.update_channel()
        self.assertEqual(self.app.channel.snr_db, 0)

    def test_fading_enabled(self):
        self.app.var_qsb.get.return_value = True
        self.app.update_channel()
        self.assertTrue(self.app.channel.fading)
        self.assertTrue(self.app.channel.multipath)

    def test_fading_disabled(self):
        self.app.var_qsb.get.return_value = False
        self.app.update_channel()
        self.assertFalse(self.app.channel.fading)
        self.assertFalse(self.app.channel.multipath)


# ======================================================================
# 4. toggle_tx
# ======================================================================
class TestToggleTx(_GUITestBase):

    def test_toggle_tx_no_image(self):
        """toggle_tx should be a no-op if no TX object loaded."""
        self.app.tx = None
        self.app.toggle_tx()
        self.assertFalse(self.app.is_transmitting)

    def test_toggle_tx_start_stop(self):
        """Start TX, then stop TX."""
        self.app.tx = self.MockTx.return_value

        # Start
        self.app.toggle_tx()
        self.assertTrue(self.app.is_transmitting)

        # Stop
        self.app.toggle_tx()
        self.assertFalse(self.app.is_transmitting)


# ======================================================================
# 5. toggle_play
# ======================================================================
class TestTogglePlay(_GUITestBase):

    def test_toggle_play_no_samples(self):
        """Play should do nothing if no audio data and no WAV file."""
        self.app.valid_samples = 0
        self.app.audio_chunks = []
        # Prevent auto-load of transmission.wav
        with patch('os.path.exists', return_value=False):
            self.app.toggle_play()
        self.assertFalse(self.app.is_playing)

    def test_toggle_play_with_data(self):
        """Play should start when audio data exists."""
        self._populate_audio(4096)
        with patch.object(gui, 'AUDIO_AVAILABLE', False), \
             patch('os.path.exists', return_value=False):
            self.app.toggle_play()
            self.assertTrue(self.app.is_playing)

            # Stop
            self.app.toggle_play()
            self.assertFalse(self.app.is_playing)

    def test_toggle_play_empty_chunks_no_crash(self):
        """Empty audio_chunks should not crash (the fix we applied)."""
        self.app.valid_samples = 100
        self.app.audio_chunks = []
        with patch('os.path.exists', return_value=False):
            # Should NOT raise, should just print warning and return
            self.app.toggle_play()
        self.assertFalse(self.app.is_playing)


# ======================================================================
# 6. save_wav
# ======================================================================
class TestSaveWav(_GUITestBase):

    def test_save_wav_no_data(self):
        """Saving with no audio data should be a no-op."""
        self.app.audio_chunks = []
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        try:
            self.app.save_wav(tmp.name)
            self.assertEqual(os.path.getsize(tmp.name), 0)
        finally:
            os.unlink(tmp.name)

    def test_save_wav_with_data(self):
        """Saving valid audio data should create a WAV file."""
        self._populate_audio(2048)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        try:
            self.app.save_wav(tmp.name)
            self.assertGreater(os.path.getsize(tmp.name), 0)
        finally:
            os.unlink(tmp.name)

    def test_save_wav_cancel_dialog(self):
        """Cancelling file dialog should be a no-op."""
        gui.filedialog.asksaveasfilename.return_value = ""
        self._populate_audio(1024)
        self.app.save_wav()


# ======================================================================
# 7. load_wav
# ======================================================================
class TestLoadWav(_GUITestBase):

    def test_load_wav_valid_file(self):
        """Load a real WAV file and verify audio_chunks are populated."""
        import scipy.io.wavfile as wavfile
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        try:
            data = np.random.rand(1024).astype(np.float32)
            wavfile.write(tmp.name, 12000, data)
            self.app.load_wav(tmp.name)
            self.assertEqual(self.app.valid_samples, 1024)
            self.assertEqual(self.app.sample_rate, 12000)
        finally:
            os.unlink(tmp.name)

    def test_load_wav_cancel_dialog(self):
        """Cancelling the file dialog should be a no-op."""
        gui.filedialog.askopenfilename.return_value = ""
        self.app.load_wav()


# ======================================================================
# 8. reset_rx
# ======================================================================
class TestResetRx(_GUITestBase):

    def test_reset_rx_creates_new_receiver(self):
        """reset_rx should create a fresh HoloRx and reset rx_pos."""
        old_rx = MagicMock()
        self.app.rx = old_rx
        self.app.rx_pos = 5000
        self.MockRx.reset_mock()
        self.app.reset_rx()
        self.assertEqual(self.app.rx_pos, 0)
        self.MockRx.assert_called_once_with(42, 640, 480)

    def test_reset_rx_closes_old(self):
        """reset_rx should close the previous receiver."""
        old_rx = MagicMock()
        self.app.rx = old_rx
        self.app.reset_rx()
        old_rx.close.assert_called_once()


# ======================================================================
# 9. on_seek
# ======================================================================
class TestOnSeek(_GUITestBase):

    @patch.object(HoloGUI, 'reset_rx')
    def test_on_seek_forward(self, mock_reset):
        """Seeking forward should update play_pos without resetting RX."""
        self.app.rx_pos = 100
        self.app.on_seek(200)
        self.assertEqual(self.app.play_pos, 200)
        mock_reset.assert_not_called()

    @patch.object(HoloGUI, 'reset_rx')
    def test_on_seek_backward_resets_rx(self, mock_reset):
        """Seeking backward should reset RX."""
        self.app.rx_pos = 500
        self.app.on_seek(100)
        self.assertEqual(self.app.play_pos, 100)
        mock_reset.assert_called_once()
        self.assertEqual(self.app.rx_pos, 0)


# ======================================================================
# 10. get_audio_slice
# ======================================================================
class TestGetAudioSlice(_GUITestBase):

    def test_empty_chunks(self):
        """No audio → empty array."""
        self.app.audio_chunks = []
        result = self.app.get_audio_slice(100)
        self.assertEqual(len(result), 0)

    def test_full_slice(self):
        """Requesting more than available → returns all."""
        chunk = np.ones(500, dtype=np.float32)
        self.app.audio_chunks = [chunk]
        self.app.valid_samples = 500
        result = self.app.get_audio_slice(1000)
        self.assertEqual(len(result), 500)

    def test_partial_slice(self):
        """Requesting a subset → returns correct length from tail."""
        chunk = np.arange(1000, dtype=np.float32)
        self.app.audio_chunks = [chunk]
        self.app.valid_samples = 1000
        result = self.app.get_audio_slice(100)
        self.assertEqual(len(result), 100)
        np.testing.assert_array_equal(result, np.arange(900, 1000, dtype=np.float32))

    def test_multi_chunk_slice(self):
        """Spanning multiple chunks still gives correct length."""
        c1 = np.ones(300, dtype=np.float32)
        c2 = np.ones(300, dtype=np.float32) * 2
        self.app.audio_chunks = [c1, c2]
        self.app.valid_samples = 600
        result = self.app.get_audio_slice(400)
        self.assertEqual(len(result), 400)


# ======================================================================
# 11. on_close
# ======================================================================
class TestOnClose(_GUITestBase):

    def test_on_close_sets_flags(self):
        """on_close should clear all running flags."""
        self.app.is_transmitting = True
        self.app.is_playing = True
        self.app.running = True
        self.app.play_stream = None
        self.app.tx = None
        self.app.rx = None

        self.app.on_close()

        self.assertFalse(self.app.running)
        self.assertFalse(self.app.is_transmitting)
        self.assertFalse(self.app.is_playing)
        self.app.root.destroy.assert_called_once()

    def test_on_close_closes_resources(self):
        """on_close should close TX, RX, and audio stream."""
        mock_tx = MagicMock()
        mock_rx = MagicMock()
        mock_stream = MagicMock()
        self.app.tx = mock_tx
        self.app.rx = mock_rx
        self.app.play_stream = mock_stream

        self.app.on_close()

        mock_tx.close.assert_called_once()
        mock_rx.close.assert_called_once()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


# ======================================================================
# 12. audio_buffering
# ======================================================================
class TestAudioBuffering(_GUITestBase):

    def test_audio_buffer_accumulation(self):
        """Audio chunks should accumulate correctly."""
        chunk = np.random.rand(1024).astype(np.float32)
        with self.app.lock:
            self.app.audio_chunks.append(chunk)
            self.app.valid_samples += len(chunk)
        self.assertEqual(self.app.valid_samples, 1024)
        self.assertEqual(len(self.app.audio_chunks), 1)

    def test_multiple_chunks_accumulate(self):
        """Multiple appends should increase totals."""
        for _ in range(5):
            chunk = np.random.rand(512).astype(np.float32)
            with self.app.lock:
                self.app.audio_chunks.append(chunk)
                self.app.valid_samples += len(chunk)
        self.assertEqual(self.app.valid_samples, 2560)
        self.assertEqual(len(self.app.audio_chunks), 5)


# ======================================================================
# 13. RadioChannel
# ======================================================================
class TestRadioChannel(_GUITestBase):

    def test_channel_bypass(self):
        """With no effects, channel.apply should return input unchanged."""
        self.app.channel.snr_db = None
        self.app.channel.fading = False
        self.app.channel.multipath = False
        data = np.ones(100, dtype=np.float32)
        result = self.app.channel.apply(data)
        np.testing.assert_array_equal(data, result)

    def test_channel_noise_changes_signal(self):
        """With noise enabled, output should differ from input."""
        self.app.channel.snr_db = 10
        data = np.ones(1000, dtype=np.float32)
        result = self.app.channel.apply(data)
        self.assertFalse(np.array_equal(data, result))

    def test_channel_fading(self):
        """With fading enabled, output should differ from input."""
        self.app.channel.fading = True
        data = np.ones(1000, dtype=np.float32)
        result = self.app.channel.apply(data)
        self.assertFalse(np.array_equal(data, result))

    def test_channel_multipath(self):
        """With multipath enabled, output should differ from input."""
        self.app.channel.multipath = True
        data = np.ones(1000, dtype=np.float32)
        result = self.app.channel.apply(data)
        self.assertFalse(np.array_equal(data, result))


if __name__ == '__main__':
    unittest.main()
