"""
Simulation runner — uses the C binary for the heavy lifting.

Downloads test images, converts to raw grayscale, calls ./holo_protocol,
and assembles comparison grids via matplotlib.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
from PIL import Image

from utils import download_image, load_image_gray, save_comparison, psnr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINARY = os.path.join(os.path.dirname(__file__), "holo_cuda")
SEED = 42
SYNC_INTERVAL = 1000
IMAGE_DIR = "test_images"
OUTPUT_DIR = "output"
RAW_DIR = "raw_tmp"

WIDTH, HEIGHT = 640, 480
NUM_PIXELS = WIDTH * HEIGHT  # 307_200

MEASUREMENT_COUNTS = [
    1_000, 5_000, 25_000, 100_000,
    NUM_PIXELS,          # 1× pixels (307,200)
    NUM_PIXELS * 2,      # 2× pixels (614,400)
]

SNR_LEVELS: list[float | None] = [None, 20.0, 10.0]

TEST_IMAGES = [
    ("https://picsum.photos/id/10/640/480", "landscape.png"),
    ("https://picsum.photos/id/22/640/480", "coffee.png"),
    ("https://picsum.photos/id/96/640/480", "architecture.png"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_binary():
    if not os.path.isfile(BINARY):
        print("Compiling holo_protocol.c ...")
        subprocess.check_call([
            "gcc", "-O3", "-march=native",
            "-o", BINARY, "holo_protocol.c", "-lm"
        ])
    print(f"  Binary: {BINARY}")


def ensure_test_images() -> list[str]:
    paths = []
    for url, name in TEST_IMAGES:
        path = os.path.join(IMAGE_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} …")
            download_image(url, path)
        else:
            print(f"  Cached: {path}")
        paths.append(path)
    return paths


def image_to_raw(png_path: str, raw_path: str) -> None:
    """Convert a PNG to raw grayscale uint8."""
    img = Image.open(png_path).convert("L").resize((WIDTH, HEIGHT),
                                                    Image.LANCZOS)
    os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(img.tobytes())


def raw_to_array(raw_path: str) -> np.ndarray:
    """Read a raw grayscale file back into a numpy array."""
    data = np.fromfile(raw_path, dtype=np.uint8)
    return data.reshape(HEIGHT, WIDTH)


def run_protocol(input_raw: str, num_meas: int,
                 snr_db: float | None, output_raw: str) -> None:
    """Call the C binary."""
    snr_arg = str(snr_db) if snr_db is not None else "-1"
    cmd = [
        BINARY, input_raw,
        str(WIDTH), str(HEIGHT), str(num_meas),
        str(SEED), str(SYNC_INTERVAL), snr_arg,
        output_raw,
    ]
    subprocess.check_call(cmd, stderr=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_simulation() -> None:
    print("=" * 60)
    print("Holographic Image Transmission — C Simulation")
    print("=" * 60)

    ensure_binary()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    print("\n▸ Downloading / loading test images…")
    image_paths = ensure_test_images()

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        original = load_image_gray(img_path)

        # Convert to raw for the C binary
        raw_in = os.path.join(RAW_DIR, f"{img_name}.raw")
        image_to_raw(img_path, raw_in)

        print(f"\n{'─' * 60}")
        print(f"Image: {img_name}  ({WIDTH}×{HEIGHT})")
        print(f"{'─' * 60}")

        for snr in SNR_LEVELS:
            snr_label = "noiseless" if snr is None else f"SNR {snr:.0f} dB"
            print(f"\n  Channel: {snr_label}")

            reconstructions = []
            labels = []

            for n_meas in MEASUREMENT_COUNTS:
                snr_tag = "clean" if snr is None else f"snr{int(snr)}"
                raw_out = os.path.join(
                    RAW_DIR, f"{img_name}_{snr_tag}_{n_meas}.raw"
                )
                run_protocol(raw_in, n_meas, snr, raw_out)

                recon = raw_to_array(raw_out)
                p = psnr(original, recon)
                ratio = n_meas / NUM_PIXELS
                label = f"{n_meas:,} ({ratio:.2f}×)"
                print(f"    {label:>22s}  PSNR={p:5.1f} dB")
                reconstructions.append(recon)
                labels.append(label)

            # Save comparison grid
            snr_file = "noiseless" if snr is None else f"snr{int(snr)}db"
            out_path = os.path.join(OUTPUT_DIR,
                                    f"{img_name}_{snr_file}.png")
            save_comparison(original, reconstructions, labels, out_path,
                            title=f"{img_name} — {snr_label}")

    print("\n" + "=" * 60)
    print(f"Done!  Results in  ./{OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
