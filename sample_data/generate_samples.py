#!/usr/bin/env python3
"""
generate_samples.py — Generate synthetic 8-bit RTL-SDR IQ sample files.

Creates a small demo dataset with two simulated devices:
  - device_A: ~50 kHz carrier offset, fast rise time
  - device_B: ~200 kHz carrier offset, slow rise time

Run from the project root:
    python sample_data/generate_samples.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

SR = 2.4e6          # 2.4 MSPS
N_TOTAL = 65536     # total samples per capture
N_BURST = 32768     # burst length
N_SAMPLES_PER_CLASS = 5

OUTPUT_DIR = Path(__file__).parent


def make_iq_uint8(
    freq_hz: float,
    ramp_len: int,
    noise_std: float = 0.05,
    amplitude: float = 0.7,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate a burst transmission and encode as 8-bit IQ (RTL-SDR format).

    Returns uint8 array [I0, Q0, I1, Q1, …] with DC at 127.5.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(N_TOTAL) / SR
    signal = np.zeros(N_TOTAL, dtype=np.complex64)

    burst_start = N_TOTAL // 4
    burst = amplitude * np.exp(1j * 2 * np.pi * freq_hz * t[burst_start:burst_start + N_BURST])

    # Apply rise/fall ramp
    ramp_up = np.minimum(np.linspace(0, 1, ramp_len), 1.0)
    ramp_down = ramp_up[::-1]
    burst[:ramp_len] *= ramp_up
    burst[-ramp_len:] *= ramp_down

    signal[burst_start:burst_start + N_BURST] = burst
    signal += (
        noise_std * rng.standard_normal(N_TOTAL) + 1j * noise_std * rng.standard_normal(N_TOTAL)
    ).astype(np.complex64)

    # Encode to uint8 (clip + scale to [0, 255], DC at 127.5)
    real_u8 = np.clip(signal.real * 127.5 + 127.5, 0, 255).astype(np.uint8)
    imag_u8 = np.clip(signal.imag * 127.5 + 127.5, 0, 255).astype(np.uint8)

    # Interleave I and Q
    output = np.empty(N_TOTAL * 2, dtype=np.uint8)
    output[0::2] = real_u8
    output[1::2] = imag_u8
    return output


def main():
    rng = np.random.default_rng(seed=42)

    devices = {
        "device_A": dict(freq_hz=50e3,  ramp_len=256,  noise_std=0.04),
        "device_B": dict(freq_hz=200e3, ramp_len=2048, noise_std=0.06),
    }

    for label, params in devices.items():
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating {N_SAMPLES_PER_CLASS} captures for {label} …")
        for i in range(N_SAMPLES_PER_CLASS):
            data = make_iq_uint8(**params, rng=rng)
            fpath = out_dir / f"capture_{i:02d}.iq"
            data.tofile(str(fpath))
            print(f"  → {fpath} ({len(data):,} bytes)")

    print("\nDone. Run the classifier:")
    print("  python rf_fingerprint.py train --data-dir sample_data --model-out model.pkl")
    print("  python rf_fingerprint.py classify --model model.pkl sample_data/device_A/capture_00.iq")


if __name__ == "__main__":
    main()
