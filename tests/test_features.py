"""
tests/test_features.py — Unit tests for features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from features import (
    load_iq,
    detect_burst,
    compute_rise_time,
    compute_frequency_offset,
    compute_phase_noise,
    compute_spectral_flatness,
    compute_burst_shape,
    extract_features,
    features_to_vector,
)


SR = 2.4e6  # default sample rate


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_cw_burst(n_total=8192, n_burst=4096, freq_hz=10e3, amplitude=1.0, noise_std=0.01):
    """Simulate a CW burst embedded in noise."""
    t = np.arange(n_total) / SR
    signal = np.zeros(n_total, dtype=np.complex64)
    burst_slice = slice(n_total // 4, n_total // 4 + n_burst)
    signal[burst_slice] = amplitude * np.exp(1j * 2 * np.pi * freq_hz * t[burst_slice])
    # Add a short ramp at the start of the burst
    ramp_len = 256
    ramp = np.linspace(0, 1, ramp_len)
    start = n_total // 4
    signal[start:start + ramp_len] *= ramp
    signal += (noise_std * np.random.randn(n_total) + 1j * noise_std * np.random.randn(n_total)).astype(np.complex64)
    return signal


# ---------------------------------------------------------------------------
# load_iq
# ---------------------------------------------------------------------------

class TestLoadIQ:
    def test_passthrough_complex_array(self):
        arr = np.array([1+2j, 3+4j], dtype=np.complex64)
        out = load_iq(arr)
        assert np.allclose(out, arr)
        assert out.dtype == np.complex64

    def test_load_uint8_raw(self, tmp_path):
        # Build a tiny RTL-SDR-like uint8 file
        iq = np.array([127, 127, 200, 50], dtype=np.uint8)
        fpath = tmp_path / "test.iq"
        iq.tofile(str(fpath))
        samples = load_iq(fpath)
        assert samples.dtype == np.complex64
        assert len(samples) == 2  # 4 bytes → 2 IQ pairs
        # DC bias removed: 127.5 subtracted
        assert abs(samples[0].real - (127 - 127.5)) < 1e-3
        assert abs(samples[0].imag - (127 - 127.5)) < 1e-3

    def test_odd_length_file_truncated(self, tmp_path):
        raw = np.arange(7, dtype=np.uint8)
        fpath = tmp_path / "odd.iq"
        raw.tofile(str(fpath))
        samples = load_iq(fpath)
        assert len(samples) == 3  # floor(7/2)


# ---------------------------------------------------------------------------
# detect_burst
# ---------------------------------------------------------------------------

class TestDetectBurst:
    def test_detects_burst(self):
        signal = make_cw_burst()
        start, end = detect_burst(signal, threshold_db=-15, min_burst_len=256)
        n_total = len(signal)
        expected_start = n_total // 4
        # Allow 10% tolerance
        assert abs(start - expected_start) < n_total * 0.1
        assert end > start

    def test_silent_signal_falls_back(self):
        signal = np.zeros(1024, dtype=np.complex64)
        start, end = detect_burst(signal, min_burst_len=1)
        assert start == 0
        assert end == len(signal)


# ---------------------------------------------------------------------------
# Rise time
# ---------------------------------------------------------------------------

class TestRiseTime:
    def test_ramp_signal(self):
        n = 4096
        ramp_len = 512
        t = np.arange(n) / SR
        signal = np.zeros(n, dtype=np.complex64)
        signal[:ramp_len] = np.linspace(0, 1, ramp_len) * np.exp(1j * 2 * np.pi * 10e3 * t[:ramp_len])
        signal[ramp_len:] = np.exp(1j * 2 * np.pi * 10e3 * t[ramp_len:])
        rt = compute_rise_time(signal, sample_rate=SR)
        expected = (ramp_len * 0.8) / SR * 1e6  # µs
        # Should be in a reasonable ballpark
        assert rt > 0

    def test_zero_signal(self):
        assert compute_rise_time(np.zeros(256, dtype=np.complex64)) == 0.0


# ---------------------------------------------------------------------------
# Frequency offset
# ---------------------------------------------------------------------------

class TestFrequencyOffset:
    def test_known_offset(self):
        target_hz = 15e3
        t = np.arange(4096) / SR
        signal = np.exp(1j * 2 * np.pi * target_hz * t).astype(np.complex64)
        est = compute_frequency_offset(signal, sample_rate=SR)
        # Allow ±5% tolerance
        assert abs(est - target_hz) < target_hz * 0.05

    def test_zero_offset(self):
        n = 4096
        # Constant phase → 0 Hz offset
        signal = np.ones(n, dtype=np.complex64)
        est = compute_frequency_offset(signal, sample_rate=SR)
        assert abs(est) < 100  # within 100 Hz of zero


# ---------------------------------------------------------------------------
# Phase noise
# ---------------------------------------------------------------------------

class TestPhaseNoise:
    def test_returns_float(self):
        t = np.arange(4096) / SR
        signal = np.exp(1j * 2 * np.pi * 10e3 * t).astype(np.complex64)
        pn = compute_phase_noise(signal, sample_rate=SR, offset_hz=1e3)
        assert isinstance(pn, float)
        assert pn <= 0  # phase noise in dBc should be ≤ 0 for a good signal

    def test_noisy_worse_than_clean(self):
        t = np.arange(8192) / SR
        clean = np.exp(1j * 2 * np.pi * 10e3 * t).astype(np.complex64)
        noisy = clean + (0.3 * np.random.randn(8192) + 1j * 0.3 * np.random.randn(8192)).astype(np.complex64)
        pn_clean = compute_phase_noise(clean, sample_rate=SR)
        pn_noisy = compute_phase_noise(noisy, sample_rate=SR)
        # Noisy signal should have worse (higher / less negative) phase noise
        assert pn_noisy >= pn_clean


# ---------------------------------------------------------------------------
# Spectral flatness
# ---------------------------------------------------------------------------

class TestSpectralFlatness:
    def test_white_noise_near_one(self):
        noise = (np.random.randn(8192) + 1j * np.random.randn(8192)).astype(np.complex64)
        flatness = compute_spectral_flatness(noise, sample_rate=SR)
        assert 0.5 < flatness <= 1.0

    def test_tone_low_flatness(self):
        t = np.arange(8192) / SR
        tone = np.exp(1j * 2 * np.pi * 100e3 * t).astype(np.complex64)
        flatness = compute_spectral_flatness(tone, sample_rate=SR)
        assert flatness < 0.5

    def test_range(self):
        signal = make_cw_burst()
        flatness = compute_spectral_flatness(signal, sample_rate=SR)
        assert 0.0 <= flatness <= 1.0


# ---------------------------------------------------------------------------
# Burst shape
# ---------------------------------------------------------------------------

class TestBurstShape:
    def test_length(self):
        signal = make_cw_burst()
        shape = compute_burst_shape(signal, n_points=16)
        assert len(shape) == 16

    def test_normalized(self):
        signal = make_cw_burst()
        shape = compute_burst_shape(signal, n_points=16)
        assert shape.max() <= 1.0 + 1e-6
        assert shape.min() >= -1e-6

    def test_zero_signal(self):
        shape = compute_burst_shape(np.zeros(256, dtype=np.complex64))
        assert np.all(shape == 0)


# ---------------------------------------------------------------------------
# extract_features + features_to_vector
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_dict_with_correct_keys(self):
        signal = make_cw_burst()
        feats = extract_features(signal, sample_rate=SR)
        expected_keys = {
            "rise_time_us", "freq_offset_hz", "phase_noise_dbc",
            "spectral_flatness", "burst_shape", "n_samples",
            "burst_start", "burst_end",
        }
        assert expected_keys == set(feats.keys())

    def test_burst_shape_shape(self):
        signal = make_cw_burst()
        feats = extract_features(signal, sample_rate=SR, burst_shape_points=16)
        assert feats["burst_shape"].shape == (16,)

    def test_n_samples_correct(self):
        signal = make_cw_burst(n_total=4096)
        feats = extract_features(signal, sample_rate=SR)
        assert feats["n_samples"] == 4096

    def test_features_to_vector_length(self):
        signal = make_cw_burst()
        feats = extract_features(signal, sample_rate=SR, burst_shape_points=16)
        vec = features_to_vector(feats)
        assert vec.shape == (4 + 16,)  # 4 scalars + 16 burst shape
        assert vec.dtype == np.float32
