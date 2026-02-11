"""
Unit tests for the holographic image transmission protocol.

Run with:  python -m pytest test_protocol.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from protocol import (
    generate_mask,
    Transmitter,
    Receiver,
    add_channel_noise,
    run_pipeline,
)
from utils import psnr


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

class TestGenerateMask:
    def test_shape(self):
        rng = np.random.default_rng(0)
        mask = generate_mask(rng, 100)
        assert mask.shape == (100,)

    def test_values_are_pm1(self):
        rng = np.random.default_rng(0)
        mask = generate_mask(rng, 10_000)
        unique = set(mask.tolist())
        assert unique == {-1, 1}

    def test_approximately_balanced(self):
        """Over many samples, roughly 50 % should be +1."""
        rng = np.random.default_rng(0)
        mask = generate_mask(rng, 100_000)
        frac_pos = np.mean(mask == 1)
        assert 0.49 < frac_pos < 0.51

    def test_reproducibility(self):
        """Same RNG seed → same mask."""
        m1 = generate_mask(np.random.default_rng(42), 500)
        m2 = generate_mask(np.random.default_rng(42), 500)
        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_differ(self):
        m1 = generate_mask(np.random.default_rng(0), 500)
        m2 = generate_mask(np.random.default_rng(1), 500)
        assert not np.array_equal(m1, m2)


# ---------------------------------------------------------------------------
# Transmitter / Receiver sync
# ---------------------------------------------------------------------------

class TestSync:
    """Verify sender and receiver PRNGs stay in lock-step."""

    def test_masks_match(self):
        """TX and RX must generate identical masks for every measurement."""
        seed, sync_interval = 99, 50
        image = np.random.default_rng(0).uniform(0, 255, (8, 8))

        tx = Transmitter(image, seed=seed, sync_interval=sync_interval)
        rx = Receiver(seed=seed, width=8, height=8,
                      sync_interval=sync_interval)

        # Step both and verify their PRNG states stay identical
        for i in range(sync_interval * 3):  # across multiple epochs
            tx_state = tx._rng.bit_generator.state
            rx_state = rx._rng.bit_generator.state
            assert tx_state == rx_state, f"PRNG desync at measurement {i}"

            val = tx.next_measurement()
            rx.receive(val)

    def test_epoch_boundary_resets_rng(self):
        """PRNG should reset when crossing a sync boundary."""
        seed, sync_interval = 7, 10
        image = np.zeros((4, 4))
        tx = Transmitter(image, seed=seed, sync_interval=sync_interval)

        # Step to the last measurement of epoch 0
        for _ in range(sync_interval - 1):
            tx.next_measurement()
        assert tx._epoch == 0

        # The next call crosses into epoch 1
        tx.next_measurement()
        assert tx._epoch == 1
        assert tx._idx_in_epoch == 0


# ---------------------------------------------------------------------------
# Noiseless reconstruction quality
# ---------------------------------------------------------------------------

class TestNoiselessReconstruction:
    """With enough measurements and no noise, the reconstruction should
    converge toward the original image."""

    @pytest.fixture
    def simple_image(self):
        """A small deterministic test image."""
        rng = np.random.default_rng(123)
        return rng.uniform(0, 255, (8, 8))

    def test_more_measurements_improves_psnr(self, simple_image):
        """PSNR should increase monotonically with measurement count."""
        counts = [50, 200, 500, 2000]
        prev_p = -np.inf
        for n in counts:
            recon = run_pipeline(simple_image, n, seed=42,
                                 sync_interval=100, snr_db=None)
            p = psnr(simple_image, recon)
            assert p >= prev_p - 1.0, (
                f"PSNR should generally improve: {prev_p:.1f} → {p:.1f} "
                f"at {n} measurements"
            )
            prev_p = p

    def test_high_measurement_count_is_recognisable(self, simple_image):
        """At n_meas = 10× pixels, PSNR should be decent (>15 dB)."""
        n_pixels = simple_image.size
        recon = run_pipeline(simple_image, n_pixels * 10, seed=42,
                             sync_interval=100, snr_db=None)
        p = psnr(simple_image, recon)
        assert p > 15.0, f"Expected PSNR > 15 dB at 10× oversampling, got {p:.1f}"


# ---------------------------------------------------------------------------
# Channel noise
# ---------------------------------------------------------------------------

class TestChannelNoise:
    def test_noise_degrades_quality(self):
        """Adding noise should reduce PSNR vs noiseless."""
        rng = np.random.default_rng(7)
        image = rng.uniform(0, 255, (8, 8))
        n_meas = 500

        recon_clean = run_pipeline(image, n_meas, seed=1, snr_db=None)
        recon_noisy = run_pipeline(image, n_meas, seed=1, snr_db=5.0)

        p_clean = psnr(image, recon_clean)
        p_noisy = psnr(image, recon_noisy)
        assert p_clean > p_noisy, (
            f"Noiseless PSNR ({p_clean:.1f}) should beat "
            f"noisy PSNR ({p_noisy:.1f})"
        )

    def test_higher_snr_means_better_quality(self):
        """SNR 30 dB should give better reconstruction than 5 dB."""
        rng = np.random.default_rng(7)
        image = rng.uniform(0, 255, (8, 8))
        n_meas = 500

        recon_high = run_pipeline(image, n_meas, seed=1, snr_db=30.0)
        recon_low = run_pipeline(image, n_meas, seed=1, snr_db=5.0)

        p_high = psnr(image, recon_high)
        p_low = psnr(image, recon_low)
        assert p_high > p_low, (
            f"High-SNR PSNR ({p_high:.1f}) should beat "
            f"low-SNR ({p_low:.1f})"
        )

    def test_add_channel_noise_power(self):
        """Verify the noise power is roughly correct."""
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 100, size=100_000)
        snr_db = 10.0
        noisy = add_channel_noise(signal, snr_db, rng=np.random.default_rng(1))
        noise = noisy - signal

        sig_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        measured_snr = 10 * np.log10(sig_power / noise_power)
        assert abs(measured_snr - snr_db) < 0.5, (
            f"Measured SNR {measured_snr:.2f} dB, expected ~{snr_db} dB"
        )


# ---------------------------------------------------------------------------
# PSNR utility
# ---------------------------------------------------------------------------

class TestPSNR:
    def test_identical_images(self):
        img = np.random.default_rng(0).integers(0, 256, (10, 10),
                                                dtype=np.uint8)
        assert psnr(img, img) == float("inf")

    def test_known_value(self):
        a = np.zeros((10, 10), dtype=np.float64)
        b = np.ones((10, 10), dtype=np.float64) * 10
        # MSE = 100, PSNR = 10*log10(255^2/100) ≈ 28.13
        p = psnr(a, b)
        assert abs(p - 28.13) < 0.1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_result(self):
        """Two runs with the same seed must produce identical output."""
        image = np.random.default_rng(0).uniform(0, 255, (8, 8))
        r1 = run_pipeline(image, 300, seed=42, snr_db=None)
        r2 = run_pipeline(image, 300, seed=42, snr_db=None)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_different_result(self):
        image = np.random.default_rng(0).uniform(0, 255, (8, 8))
        r1 = run_pipeline(image, 300, seed=42, snr_db=None)
        r2 = run_pipeline(image, 300, seed=99, snr_db=None)
        assert not np.array_equal(r1, r2)
