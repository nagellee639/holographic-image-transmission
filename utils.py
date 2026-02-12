"""
Utility helpers for image loading, saving, and quality metrics.
"""

from __future__ import annotations

import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def download_image(url: str, path: str) -> str:
    """Download an image from *url*, resize to 640×480, save to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("L")  # grayscale
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    img.save(path)
    print(f"  Saved {path}  ({TARGET_WIDTH}×{TARGET_HEIGHT} grayscale)")
    return path


def load_image_gray(path: str) -> np.ndarray:
    """Load an image as a float64 grayscale array (H, W) in [0, 255]."""
    img = Image.open(path).convert("L")
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two images."""
    orig = original.astype(np.float64)
    recon = reconstructed.astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_metric(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two images.
    Input images should be (H, W) or (H, W, C) in [0, 255].
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Ensure float range [0, 255]? SKImage handles it if data_range is specified.
    # We should cast to match types, but ssim handles arrays.
    # Important: specify data_range=255 for correct scaling.
    return ssim(original, reconstructed, data_range=255.0)


def save_comparison(original: np.ndarray,
                    reconstructions: list[np.ndarray],
                    labels: list[str],
                    path: str,
                    title: str = "") -> None:
    """Save a side-by-side comparison figure: original + reconstructions."""
    n = 1 + len(reconstructions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for ax, img, label in zip(axes[1:], reconstructions, labels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        p = psnr(original, img)
        s = ssim_metric(original, img)
        ax.set_title(f"{label}\nPSNR {p:.1f} dB\nSSIM {s:.3f}", fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison → {path}")
