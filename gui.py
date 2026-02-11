
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import time
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except OSError:
    sd = None
    AUDIO_AVAILABLE = False
import scipy.io.wavfile as wavfile
import os
from holo_api import HoloTx, HoloRx

# Radio Modes: Name -> Sample Rate (Hz)
MODES = {
    "CW (100 Hz)": 100,
    "SSB (3 kHz)": 3000,
    "FM (12 kHz)": 12000
}

class HoloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Holographic Image Transmission")
        self.root.geometry("1400x700")

        # System State
        self.tx = None
        self.rx = None
        self.audio_chunks = [] # List of np.ndarray (float32)
        self.valid_samples = 0
        self.sample_rate = 12000
        
        self.is_transmitting = False
        self.is_playing = False
        self.play_stream = None
        self.play_pos = 0 # absolute sample index
        self.rx_pos = 0   # absolute sample index processed by RX
        self.audio_buffer = None # Flattened array for playback
        
        self.lock = threading.Lock()
        self.running = True # Global run flag
        
        # Metrics
        self.tx_samples_last = 0
        self.rx_samples_last = 0
        self.last_metrics_time = time.time()
        
        # --- UI Layout ---
        
        # --- UI Layout ---
        style = ttk.Style()
        style.theme_use('clam')
        
        container = ttk.Frame(root)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 3 Columns
        self.pane_tx = ttk.LabelFrame(container, text="1. Transmission (Source)")
        self.pane_tx.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.pane_audio = ttk.LabelFrame(container, text="2. Channel (Audio)")
        self.pane_audio.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.pane_rx = ttk.LabelFrame(container, text="3. Reconstruction (Receiver)")
        self.pane_rx.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self._init_tx_ui()
        self._init_audio_ui()
        self._init_rx_ui()
        
        # Update timer
        self.root.after(100, self.ui_update_loop)
        
        # Clean exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        print("[DEBUG] Closing application...")
        self.running = False
        self.is_transmitting = False
        self.is_playing = False
        
        if self.play_stream:
            self.play_stream.stop()
            self.play_stream.close()
            
        if self.tx: self.tx.close()
        if self.rx: self.rx.close()
        
        self.root.destroy()
        print("[DEBUG] Application closed.")

    def _init_tx_ui(self):
        f = self.pane_tx
        
        # Canvas
        self.cv_tx = tk.Canvas(f, bg="#111", width=320, height=240)
        self.cv_tx.pack(pady=10)
        
        btn = ttk.Button(f, text="Load Image", command=self.load_image)
        btn.pack(pady=5)

        btn_def = ttk.Button(f, text="Load Default (Test)", command=lambda: self.load_image("test_images/landscape.png"))
        btn_def.pack(pady=5)
        
        sep = ttk.Separator(f, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=10, padx=20)
        
        lbl = ttk.Label(f, text="Radio Bandwidth:")
        lbl.pack()
        
        self.var_mode = tk.StringVar(value="FM (12 kHz)")
        for m in MODES:
            r = ttk.Radiobutton(f, text=m, variable=self.var_mode, value=m, command=self.set_mode)
            r.pack()
            
        self.btn_tx = ttk.Button(f, text="Start Encoding (TX)", command=self.toggle_tx)
        self.btn_tx.pack(pady=20)
        
        self.lbl_tx_status = ttk.Label(f, text="No Image")
        self.lbl_tx_status.pack()
        
        self.lbl_tx_rate = ttk.Label(f, text="TX Rate: 0 S/s")
        self.lbl_tx_rate.pack()

    def _init_audio_ui(self):
        f = self.pane_audio
        
        # File Selection
        frame_file = ttk.Frame(f)
        frame_file.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame_file, text="Target File:").pack(side=tk.LEFT)
        self.var_file = tk.StringVar(value="transmission.wav")
        ttk.Entry(frame_file, textvariable=self.var_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Visualizer (Buffer state)
        self.cv_audio = tk.Canvas(f, bg="#222", width=380, height=80)
        self.cv_audio.pack(pady=20)
        self.bar_fill = self.cv_audio.create_rectangle(0,0,0,80, fill="#0f0", outline="")
        self.line_play = self.cv_audio.create_line(0,0,0,80, fill="#f00", width=2)
        
        self.lbl_time = ttk.Label(f, text="0.0s")
        self.lbl_time.pack()
        
        self.sc_seek = ttk.Scale(f, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_seek)
        self.sc_seek.pack(fill=tk.X, padx=20, pady=10)
        
        frame_btns = ttk.Frame(f)
        frame_btns.pack(pady=10)
        
        self.btn_play = ttk.Button(frame_btns, text="Start Decoding (RX/Play)", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        btn_save = ttk.Button(frame_btns, text="Save WAV", command=self.save_wav)
        btn_save.pack(side=tk.LEFT, padx=5)
        
        btn_load = ttk.Button(frame_btns, text="Load WAV", command=self.load_wav)
        btn_load.pack(side=tk.LEFT, padx=5)

    def _init_rx_ui(self):
        f = self.pane_rx
        
        self.cv_rx = tk.Canvas(f, bg="#111", width=320, height=240)
        self.cv_rx.pack(pady=10)
        
        self.lbl_rx = ttk.Label(f, text="Measurements: 0")
        self.lbl_rx.pack(pady=5)
        
        self.lbl_rx_rate = ttk.Label(f, text="RX Rate: 0 S/s")
        self.lbl_rx_rate.pack()
        
        btn = ttk.Button(f, text="Reset Receiver", command=self.reset_rx)
        btn.pack(pady=10)
        
        self.var_live = tk.BooleanVar(value=True)
        c = ttk.Checkbutton(f, text="Live Decode", variable=self.var_live)
        c.pack()
        
        self.chunk_size = 2048

    def get_audio_slice(self, length):
        """Efficiently get last 'length' samples from audio_chunks"""
        if not self.audio_chunks: return np.array([], dtype=np.float32)
        
        # If length is huge, just copy all?
        if length >= self.valid_samples:
            return np.concatenate(self.audio_chunks)
            
        # Optimization: only copy last few chunks
        needed = length
        collected = []
        
        # Iterate backwards
        for chunk in reversed(self.audio_chunks):
            chunk_len = len(chunk)
            if chunk_len >= needed:
                collected.append(chunk[-needed:])
                needed = 0
                break
            else:
                collected.append(chunk)
                needed -= chunk_len
        
        # Combine
        return np.concatenate(list(reversed(collected)))

    # --- Logic ---

    def load_image(self, p=None):
        if not p:
            print("[DEBUG] Opening file dialog...")
            p = filedialog.askopenfilename()
        
        if not p: return
        try:
            print(f"[DEBUG] Loading image: {p}")
            pil = Image.open(p).convert("L").resize((640,480), Image.LANCZOS)
            self.tx_img = np.asarray(pil)
            
            # Show
            disp = pil.resize((320,240))
            self.tx_ph = ImageTk.PhotoImage(disp)
            self.cv_tx.create_image(160,120, image=self.tx_ph)
            
            # Setup TX
            if self.tx: 
                print("[DEBUG] Closing existing TX")
                self.tx.close()
            
            print("[DEBUG] Creating HoloTx...")
            self.tx = HoloTx(42, 640, 480, self.tx_img)
            print("[DEBUG] HoloTx created. Resetting RX...")
            self.reset_rx()
            print("[DEBUG] RX reset done.")
            
            self.lbl_tx_status.config(text="Ready")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_mode(self):
        self.sample_rate = MODES[self.var_mode.get()]
        self.lbl_tx_status.config(text=f"Rate: {self.sample_rate} Hz")

    def toggle_tx(self):
        if not self.tx: return
        if self.is_transmitting:
            print("[DEBUG] Stopping Transmission...")
            self.is_transmitting = False
            self.btn_tx.config(text="Start Encoding (TX)")
            
            # Auto-save on stop
            self.save_wav(self.var_file.get())
        else:
            print("[DEBUG] Starting Transmission...")
            
            # CRITICAL FIX: Reset TX to ensure PRNG starts from seed 42
            # This matches the RX which resets on Play start.
            if hasattr(self, 'tx_img'):
                if self.tx: self.tx.close()
                print("[DEBUG] Re-creating HoloTx for sync...")
                self.tx = HoloTx(42, 640, 480, self.tx_img)
            
            # Clear buffer for new transmission
            with self.lock:
                self.audio_chunks = []
                self.valid_samples = 0
                self.play_pos = 0
                self.rx_pos = 0
            
            self.is_transmitting = True
            self.btn_tx.config(text="Stop Encoding (TX)")
            threading.Thread(target=self.tx_thread, daemon=True).start()

    def tx_thread(self):
        print("[DEBUG] TX Thread Started")
        chunk = 2048
        try:
            while self.is_transmitting and self.running:
                t0 = time.time()
                # print("[DEBUG] Generating chunk...")
                data = self.tx.generate(chunk) # float32
                
                with self.lock:
                    self.audio_chunks.append(data)
                    self.valid_samples += len(data)
                
                dt = chunk / self.sample_rate
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except Exception as e:
            print(f"[ERROR] TX Thread Crashed: {e}")
            import traceback
            traceback.print_exc()
            self.is_transmitting = False
        print("[DEBUG] TX Thread Ended")

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            if self.play_stream:
                self.play_stream.stop()
                self.play_stream.close()
                self.play_stream = None
            self.btn_play.config(text="Start Decoding (RX/Play)")
        else:
            # Auto-load on play
            target_file = self.var_file.get()
            if os.path.exists(target_file):
                print(f"[DEBUG] Auto-loading {target_file} for playback...")
                self.load_wav(target_file)
            
            if self.valid_samples == 0: 
                print("[WARN] No samples to play")
                return
            
            print(f"[DEBUG] Playing {self.valid_samples} samples. Audio Avail: {AUDIO_AVAILABLE}")
            
            # Prepare buffer for playback (flatten once)
            # This simplifies callback logic immensely
            try:
                self.audio_buffer = np.concatenate(self.audio_chunks)
            except Exception as e:
                print(f"[ERROR] Failed to flatten audio: {e}")
                return
                
            self.is_playing = True
            self.btn_play.config(text="Pause Decoding")
            
            # If we are at end, restart
            if self.play_pos >= self.valid_samples:
                self.play_pos = 0
                self.rx_pos = 0
                if self.rx: self.reset_rx()
            
            if AUDIO_AVAILABLE:
                try:
                    self.play_stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        callback=self.audio_callback,
                        dtype='float32'
                    )
                    self.play_stream.start()
                    print("[DEBUG] Audio Stream Started")
                except Exception as e:
                    print(f"[ERROR] Stream start failed: {e}")
                    messagebox.showerror("Audio Error", f"Could not open audio device: {e}\nPlaying silently for simulation.")
                    self.is_playing = True
                    threading.Thread(target=self.pseudo_play_thread, daemon=True).start()
            else:
                # Fallback: Pseudo-play without sound
                print("[DEBUG] Audio unavailable. Starting Pseudo-Play.")
                self.is_playing = True
                threading.Thread(target=self.pseudo_play_thread, daemon=True).start()

    def pseudo_play_thread(self):
        """Simulate playback timing if no audio device"""
        print("[DEBUG] Pseudo-Play Thread Started")
        chunk = 1024
        while self.is_playing and self.running:
            time.sleep(chunk / self.sample_rate)
            self.play_pos += chunk
            if self.play_pos >= self.valid_samples:
                self.is_playing = False
                break
            # report pos occasionally
            if self.play_pos % 5000 < chunk:
                print(f"[DEBUG] Pseudo-play pos: {self.play_pos}/{self.valid_samples}")

    def audio_callback(self, outdata, frames, time_info, status):
        # Read from self.audio_buffer[self.play_pos : self.play_pos + frames]
        if status:
            print(f"[WARN] Audio Callback Status: {status}")
        
        # Throttled debug print
        if self.play_pos % 24000 < frames:
             print(f"[DEBUG] Audio Callback running at pos {self.play_pos}")

        if self.audio_buffer is None:
            outdata.fill(0)
            return

        chunk_size = len(outdata)
        remaining = len(self.audio_buffer) - self.play_pos
        
        if remaining <= 0:
            outdata.fill(0)
            raise sd.CallbackStop()
            
        if remaining < chunk_size:
            outdata[:remaining, 0] = self.audio_buffer[self.play_pos:]
            outdata[remaining:, 0] = 0
            self.play_pos += remaining
            raise sd.CallbackStop()
        else:
            outdata[:, 0] = self.audio_buffer[self.play_pos : self.play_pos + chunk_size]
            self.play_pos += chunk_size

    def save_wav(self, filename=None):
        if not filename:
            filename = filedialog.asksaveasfilename(defaultextension=".wav")
        if not filename: return
        
        with self.lock:
            if not self.audio_chunks: 
                print("[WARN] No chunks to save")
                return
            # Normalize to -1..1 for WAV compatibility?
            # Or just save float32
            full = np.concatenate(self.audio_chunks)
        
        try:
            wavfile.write(filename, self.sample_rate, full)
            print(f"[INFO] Saved {len(full)} samples to {filename}")
        except Exception as e:
            messagebox.showerror("Error Saving", str(e))

    def load_wav(self, filename=None):
        if not filename:
            filename = filedialog.askopenfilename()
        if not filename: return
        try:
            rate, data = wavfile.read(filename)
            self.sample_rate = rate # Update rate to match file
            if data.ndim > 1: data = data[:,0] # Mono
            
            # Normalize to float32
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            
            with self.lock:
                self.audio_chunks = [data]
                self.valid_samples = len(data)
                self.play_pos = 0
                self.rx_pos = 0
                
            self.reset_rx()
            self.lbl_tx_status.config(text=f"Loaded WAV: {rate}Hz")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_rx(self):
        if self.rx: self.rx.close()
        # Seed 42 matches TX
        self.rx = HoloRx(42, 640, 480)
        self.rx_pos = 0
        self.cv_rx.delete("all")
        self.lbl_rx.config(text="Measurements: 0")

    def on_seek(self, val):
        self.play_pos = int(float(val))
        # Reset RX if we sought backwards?
        if self.play_pos < self.rx_pos:
            self.reset_rx()
            self.rx_pos = 0 # Will catch up from 0 to play_pos

    def ui_update_loop(self):
        try:
            # Update Audio UI
            curr = self.play_pos
            tot = self.valid_samples
            rate = float(self.sample_rate)
            
            self.lbl_time.config(text=f"{curr/rate:.1f}s / {tot/rate:.1f}s")
            
            # Update Slider
            self.sc_seek.config(to=tot)
            if not self.is_playing: # Don't fight drag
                self.sc_seek.set(curr)
                
            # Draw Visualizer
            w = 380
            if tot > 0:
                x = (curr / tot) * w
                self.cv_audio.coords(self.line_play, x, 0, x, 80)
                self.cv_audio.coords(self.bar_fill, 0, 0, (self.valid_samples/tot)*w, 80)
            
            # --- Metrics ---
            now = time.time()
            dt = now - self.last_metrics_time
            if dt > 1.0:
                # TX Rate
                tx_diff = self.valid_samples - self.tx_samples_last
                tx_rate = tx_diff / dt
                self.lbl_tx_rate.config(text=f"TX Rate: {int(tx_rate)} S/s")
                self.tx_samples_last = self.valid_samples
                
                # RX Rate
                rx_diff = self.rx.measurements_count - self.rx_samples_last if self.rx else 0
                rx_rate = rx_diff / dt
                self.lbl_rx_rate.config(text=f"RX Rate: {int(rx_rate)} S/s")
                self.rx_samples_last = self.rx.measurements_count if self.rx else 0
                
                self.last_metrics_time = now
            
            # --- Live RX Logic ---
            # If Playing, target = play_pos.
            # If Transmitting and NOT playing, target = valid_samples (monitor live)
            # If Stopped, target = rx_pos (do nothing)
            
            
            target = self.rx_pos
            if self.is_playing:
                target = self.play_pos
            # elif self.is_transmitting and self.var_live.get():
            #     target = self.valid_samples # Disabled auto-RX on TX per user request
            
            if self.var_live.get() and target > self.rx_pos:
                with self.lock:
                    # Efficient fetch using helper
                    needed = target - self.rx_pos
                    # pending = self.get_audio_slice(needed) # This gets LAST needed?
                    # No, get_audio_slice gets the tail.
                    
                    # If is_transmitting (appending), the 'needed' samples ARE the tail if we are up to date!
                    # If playing back history, get_audio_slice(needed) might return wrong data if play_pos < valid_samples
                    
                    # Correction:
                    # If Monitoring Live (target == valid_samples), get_audio_slice(needed) IS correct.
                    # If Playing history (target < valid_samples), get_audio_slice returns future data.
                    
                    # So:
                    pending = None
                    if self.is_transmitting and not self.is_playing:
                        # We are at the tip.
                        if needed > 0:
                             pending = self.get_audio_slice(needed)
                    else:
                        # History playback. Fallback to concat or simple indexing if small.
                        # If large history playback is needed, we need a better structure.
                        # For now, let's just concat if we are playing history.
                        # Most likely user monitors live.
                        if self.audio_chunks:
                            full = np.concatenate(self.audio_chunks)
                            if target <= len(full):
                                pending = full[self.rx_pos : target]
                
                if pending is not None and len(pending) > 0:
                    try:
                        self.rx.accumulate(pending)
                        self.rx_pos = target
                        
                        # Update Image
                        if self.is_playing or self.is_transmitting or self.rx_pos % 5000 == 0:
                             img = self.rx.get_image()
                             pil = Image.fromarray(img)
                             disp = pil.resize((320,240))
                             self.rx_ph = ImageTk.PhotoImage(disp)
                             self.cv_rx.create_image(160,120, image=self.rx_ph)
                             self.lbl_rx.config(text=f"Measurements: {self.rx.measurements_count}")
                    except Exception as e:
                        print(f"[ERROR] RX update failed: {e}")

        except Exception as e:
            print(f"[ERROR] ui_update_loop: {e}")
            import traceback
            traceback.print_exc()

        self.root.after(100, self.ui_update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = HoloGUI(root)
    root.mainloop()
