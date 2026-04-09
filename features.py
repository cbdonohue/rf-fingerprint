"""
features.py — RF fingerprint feature extraction from IQ samples.

Supports numpy arrays and raw 8-bit RTL-SDR .iq files.
Extracts: rise time, frequency offset, phase noise, spectral flatness, burst shape.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_iq(source: Union[str, Path, np.ndarray], sample_rate: float = 2.4e6) -> np.ndarray:
    """
    Load IQ samples as a complex64 numpy array.

    Parameters
    ----------
    source : str | Path | np.ndarray
        - Path to a raw 8-bit RTL-SDR capture (.iq / .bin / .raw):
          interleaved uint8 values [I0, Q0, I1, Q1, …], DC-biased at 127.
        - Or a numpy array of complex samples (any dtype, will be cast).
    sample_rate : float
        Samples per second (used for frequency calculations).

    Returns
    -------
    np.ndarray  complex64
    """
    if isinstance(source, (str, Path)):
        raw = np.fromfile(source, dtype=np.uint8)
        if raw.size % 2 != 0:
            raw = raw[:-1]
        iq = raw.astype(np.float32) - 127.5
        samples = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)
    else:
        samples = np.asarray(source, dtype=np.complex64)
    return samples


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------

def detect_burst(
    samples: np.ndarray,
    threshold_db: float = -20.0,
    min_burst_len: int = 512,
) -> tuple[int, int]:
    """
    Return (start, end) sample indices of the first detected burst.

    Uses a simple power threshold: samples whose power exceeds
    `threshold_db` dB above the noise floor (median power).
    """
    power = np.abs(samples) ** 2
    noise_floor = np.median(power)
    if noise_floor == 0:
        noise_floor = 1e-12
    threshold_linear = noise_floor * (10 ** (threshold_db / 10))
    above = power > threshold_linear

    # Find first run of True values longer than min_burst_len
    in_burst = False
    start = 0
    for i, val in enumerate(above):
        if val and not in_burst:
            start = i
            in_burst = True
        elif not val and in_burst:
            if i - start >= min_burst_len:
                return start, i
            in_burst = False
    if in_burst and len(samples) - start >= min_burst_len:
        return start, len(samples)
    # Fallback: return entire signal
    return 0, len(samples)


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def compute_rise_time(
    samples: np.ndarray,
    sample_rate: float = 2.4e6,
    percentile_low: float = 10.0,
    percentile_high: float = 90.0,
) -> float:
    """
    Estimate burst rise time (seconds) from the envelope.

    Measures time for the envelope to rise from `percentile_low`% to
    `percentile_high`% of its peak during the leading edge of the burst.
    """
    envelope = np.abs(samples)
    peak = envelope.max()
    if peak == 0:
        return 0.0
    norm = envelope / peak
    lo = percentile_low / 100.0
    hi = percentile_high / 100.0

    # Locate leading edge: first sample above lo
    idx_lo = np.argmax(norm >= lo)
    # First sample above hi after idx_lo
    above_hi = np.where(norm[idx_lo:] >= hi)[0]
    if len(above_hi) == 0:
        return 0.0
    idx_hi = idx_lo + above_hi[0]
    rise_samples = max(idx_hi - idx_lo, 1)
    return rise_samples / sample_rate


def compute_frequency_offset(
    samples: np.ndarray,
    sample_rate: float = 2.4e6,
) -> float:
    """
    Estimate carrier frequency offset (Hz) via the instantaneous phase slope.

    Uses a linear fit to the unwrapped phase of the signal (works best on
    constant-envelope or CW signals; gives the dominant tone offset otherwise).
    """
    phase = np.unwrap(np.angle(samples))
    t = np.arange(len(phase)) / sample_rate
    if len(t) < 2:
        return 0.0
    # Linear regression: phase = 2π·f·t + φ₀  →  f = slope / (2π)
    coeffs = np.polyfit(t, phase, 1)
    freq_offset = coeffs[0] / (2 * np.pi)
    return float(freq_offset)


def compute_phase_noise(
    samples: np.ndarray,
    sample_rate: float = 2.4e6,
    offset_hz: float = 1e3,
) -> float:
    """
    Estimate phase noise (dBc/Hz) at a given offset from the carrier.

    Returns the spectral power density at `offset_hz` relative to the
    carrier peak, normalized per Hz.
    """
    n = len(samples)
    if n < 64:
        return 0.0
    # Remove mean frequency to center the carrier
    phase = np.unwrap(np.angle(samples))
    t = np.arange(n) / sample_rate
    lin = np.polyfit(t, phase, 1)
    phase_detrended = phase - np.polyval(lin, t)

    # PSD of the detrended phase (Welch-like single-window FFT)
    window = np.hanning(n)
    fft = np.fft.rfft(phase_detrended * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    psd = (np.abs(fft) ** 2) / (sample_rate * np.sum(window ** 2))

    # Carrier "power" = DC bin of the IQ spectrum
    iq_fft = np.fft.fft(samples * window)
    iq_psd = (np.abs(iq_fft) ** 2) / (sample_rate * np.sum(window ** 2))
    carrier_power = iq_psd.max()

    # Find PSD at requested offset
    idx = int(round(offset_hz / (sample_rate / n)))
    idx = min(max(idx, 1), len(psd) - 1)
    phase_noise_lin = psd[idx] / max(carrier_power, 1e-30)
    phase_noise_dbc = 10 * np.log10(max(phase_noise_lin, 1e-30))
    return float(phase_noise_dbc)


def compute_spectral_flatness(
    samples: np.ndarray,
    sample_rate: float = 2.4e6,
) -> float:
    """
    Compute spectral flatness (Wiener entropy) of the power spectrum.

    Returns a value in [0, 1]:
      - 1.0 → perfectly flat (white noise)
      - ~0   → tonal / narrow-band signal
    """
    n = len(samples)
    if n < 2:
        return 0.0
    window = np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(samples * window)) ** 2
    fft_mag = np.maximum(fft_mag, 1e-30)
    geometric_mean = np.exp(np.mean(np.log(fft_mag)))
    arithmetic_mean = np.mean(fft_mag)
    return float(geometric_mean / arithmetic_mean)


def compute_burst_shape(
    samples: np.ndarray,
    n_points: int = 16,
) -> np.ndarray:
    """
    Return a normalized burst envelope shape sampled at `n_points` equally
    spaced positions.  This captures the characteristic "silhouette" of a
    transmission (ramp-up, flat top, ramp-down, etc.).
    """
    envelope = np.abs(samples)
    if envelope.max() == 0:
        return np.zeros(n_points, dtype=np.float32)
    envelope = envelope / envelope.max()
    indices = np.linspace(0, len(envelope) - 1, n_points).astype(int)
    return envelope[indices].astype(np.float32)


# ---------------------------------------------------------------------------
# High-level feature vector
# ---------------------------------------------------------------------------

def extract_features(
    source: Union[str, Path, np.ndarray],
    sample_rate: float = 2.4e6,
    burst_threshold_db: float = -20.0,
    min_burst_len: int = 512,
    phase_noise_offset_hz: float = 1e3,
    burst_shape_points: int = 16,
) -> dict:
    """
    Extract all RF fingerprint features from an IQ source.

    Parameters
    ----------
    source : path or numpy array of complex IQ samples
    sample_rate : Hz
    burst_threshold_db : dB above noise floor to detect burst start/end
    min_burst_len : minimum burst length in samples
    phase_noise_offset_hz : offset for phase-noise measurement
    burst_shape_points : number of envelope samples for burst shape feature

    Returns
    -------
    dict with keys:
        rise_time_us       – burst rise time in microseconds
        freq_offset_hz     – carrier frequency offset in Hz
        phase_noise_dbc    – phase noise at offset, in dBc/Hz
        spectral_flatness  – Wiener entropy [0, 1]
        burst_shape        – ndarray of shape (burst_shape_points,)
        n_samples          – total samples loaded
        burst_start        – burst start sample index
        burst_end          – burst end sample index
    """
    samples = load_iq(source, sample_rate=sample_rate)
    burst_start, burst_end = detect_burst(
        samples,
        threshold_db=burst_threshold_db,
        min_burst_len=min_burst_len,
    )
    burst = samples[burst_start:burst_end]
    if len(burst) == 0:
        burst = samples  # fallback

    rise_time = compute_rise_time(burst, sample_rate=sample_rate)
    freq_offset = compute_frequency_offset(burst, sample_rate=sample_rate)
    phase_noise = compute_phase_noise(
        burst, sample_rate=sample_rate, offset_hz=phase_noise_offset_hz
    )
    flatness = compute_spectral_flatness(burst, sample_rate=sample_rate)
    shape = compute_burst_shape(burst, n_points=burst_shape_points)

    return {
        "rise_time_us": rise_time * 1e6,
        "freq_offset_hz": freq_offset,
        "phase_noise_dbc": phase_noise,
        "spectral_flatness": flatness,
        "burst_shape": shape,
        "n_samples": len(samples),
        "burst_start": burst_start,
        "burst_end": burst_end,
    }


def features_to_vector(feature_dict: dict) -> np.ndarray:
    """
    Flatten a feature dict (from extract_features) into a 1-D float32 vector
    suitable for scikit-learn classifiers.

    Layout:
        [rise_time_us, freq_offset_hz, phase_noise_dbc, spectral_flatness,
         burst_shape[0], …, burst_shape[N-1]]
    """
    scalar_keys = ["rise_time_us", "freq_offset_hz", "phase_noise_dbc", "spectral_flatness"]
    scalars = np.array([feature_dict[k] for k in scalar_keys], dtype=np.float32)
    shape_vec = feature_dict["burst_shape"].astype(np.float32)
    return np.concatenate([scalars, shape_vec])
