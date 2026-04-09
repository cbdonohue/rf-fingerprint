"""
visualize.py — Visualization utilities for RF fingerprinting.

Functions:
  plot_constellation   – IQ scatter plot
  plot_power_spectrum  – FFT magnitude spectrum
  plot_burst_envelope  – time-domain amplitude envelope
  plot_burst_shape     – normalized burst shape feature vector
  plot_all             – 2×2 summary figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; call plt.show() if you want GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from features import load_iq, detect_burst, extract_features


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_constellation(
    samples: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "IQ Constellation",
    max_points: int = 5000,
    alpha: float = 0.3,
    color: str = "#1f77b4",
) -> plt.Axes:
    """
    Plot IQ constellation (scatter of real vs imaginary part).

    Parameters
    ----------
    samples   : complex IQ samples
    ax        : existing Axes to draw on (creates figure if None)
    title     : plot title
    max_points: subsample to this many points for readability
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    if len(samples) > max_points:
        idx = np.random.choice(len(samples), max_points, replace=False)
        samples = samples[idx]
    ax.scatter(samples.real, samples.imag, s=1, alpha=alpha, color=color, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("I (In-phase)")
    ax.set_ylabel("Q (Quadrature)")
    ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    return ax


def plot_power_spectrum(
    samples: np.ndarray,
    sample_rate: float = 2.4e6,
    ax: Optional[plt.Axes] = None,
    title: str = "Power Spectrum",
    color: str = "#d62728",
    db_range: float = 60.0,
) -> plt.Axes:
    """
    Plot the FFT power spectrum in dB.

    Parameters
    ----------
    samples     : complex IQ samples
    sample_rate : Hz (used to label frequency axis)
    ax          : existing Axes to draw on
    db_range    : vertical dynamic range shown
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    n = len(samples)
    window = np.hanning(n)
    fft_vals = np.fft.fftshift(np.fft.fft(samples * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    power_db = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-12)
    peak = power_db.max()
    ax.plot(freqs / 1e3, power_db, lw=0.7, color=color)
    ax.set_ylim(peak - db_range, peak + 5)
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_burst_envelope(
    samples: np.ndarray,
    sample_rate: float = 2.4e6,
    ax: Optional[plt.Axes] = None,
    title: str = "Burst Envelope",
    color: str = "#2ca02c",
    show_burst_markers: bool = True,
    burst_threshold_db: float = -20.0,
) -> plt.Axes:
    """
    Plot the time-domain amplitude envelope of IQ samples.

    Optionally marks the detected burst region with vertical dashed lines.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    t_us = np.arange(len(samples)) / sample_rate * 1e6
    envelope = np.abs(samples)
    ax.plot(t_us, envelope, lw=0.8, color=color, label="Envelope")
    if show_burst_markers:
        start, end = detect_burst(
            samples,
            threshold_db=burst_threshold_db,
            min_burst_len=max(64, len(samples) // 200),
        )
        ax.axvline(t_us[start], color="orange", ls="--", lw=1.2, label="Burst start")
        if end < len(samples):
            ax.axvline(t_us[end - 1], color="red", ls="--", lw=1.2, label="Burst end")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_burst_shape(
    burst_shape: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Burst Shape Feature",
    color: str = "#9467bd",
) -> plt.Axes:
    """
    Plot the normalized burst-shape feature vector.

    Parameters
    ----------
    burst_shape : 1-D array (output of compute_burst_shape)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    x = np.linspace(0, 100, len(burst_shape))
    ax.plot(x, burst_shape, "o-", lw=1.5, ms=4, color=color)
    ax.fill_between(x, burst_shape, alpha=0.2, color=color)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Burst position (%)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def plot_all(
    source: Union[str, Path, np.ndarray],
    sample_rate: float = 2.4e6,
    label: str = "",
    burst_threshold_db: float = -20.0,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Generate a 2×2 summary figure for an IQ capture.

    Panels:
      TL: IQ constellation
      TR: Power spectrum
      BL: Burst envelope
      BR: Burst shape feature

    Parameters
    ----------
    source     : file path or numpy complex array
    sample_rate: Hz
    label      : optional transmitter label for figure title
    save_path  : if given, save figure to this path (PNG/PDF/SVG)
    show       : call plt.show() (requires interactive backend)

    Returns
    -------
    matplotlib Figure
    """
    from features import load_iq, detect_burst, extract_features

    samples = load_iq(source, sample_rate=sample_rate)
    feats = extract_features(
        samples,
        sample_rate=sample_rate,
        burst_threshold_db=burst_threshold_db,
    )
    burst_start = feats["burst_start"]
    burst_end = feats["burst_end"]
    burst = samples[burst_start:burst_end]
    if len(burst) == 0:
        burst = samples

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    plot_constellation(burst, ax=ax_tl)
    plot_power_spectrum(burst, sample_rate=sample_rate, ax=ax_tr)
    plot_burst_envelope(
        samples,
        sample_rate=sample_rate,
        ax=ax_bl,
        burst_threshold_db=burst_threshold_db,
    )
    plot_burst_shape(feats["burst_shape"], ax=ax_br)

    suptitle = "RF Fingerprint Analysis"
    if label:
        suptitle += f" — {label}"
    suptitle += (
        f"\n↑{feats['rise_time_us']:.2f} µs  |  Δf {feats['freq_offset_hz']:.1f} Hz  |  "
        f"PN {feats['phase_noise_dbc']:.1f} dBc/Hz  |  Flat {feats['spectral_flatness']:.3f}"
    )
    fig.suptitle(suptitle, fontsize=11, y=1.01)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Feature importance bar chart (post-training)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    fingerprinter,  # RFFingerprinter with RandomForest
    burst_shape_points: int = 16,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """
    Bar chart of feature importances (Random Forest only).

    Parameters
    ----------
    fingerprinter    : trained RFFingerprinter with model_type="rf"
    burst_shape_points: must match the value used during training
    """
    import matplotlib.cm as cm

    model = fingerprinter.model
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Feature importances only available for RandomForest model.")

    importances = model.feature_importances_
    scalar_names = ["rise_time_us", "freq_offset_hz", "phase_noise_dbc", "spectral_flatness"]
    shape_names = [f"burst_shape[{i}]" for i in range(burst_shape_points)]
    feature_names = scalar_names + shape_names

    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in idx]
    sorted_imp = importances[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    colors = cm.viridis(np.linspace(0.2, 0.9, len(sorted_imp)))
    ax.bar(range(len(sorted_imp)), sorted_imp, color=colors)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance")
    ax.set_title("RF Feature Importances")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Feature importance plot saved to {save_path}")
    return ax
