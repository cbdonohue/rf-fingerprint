# rf-fingerprint

> RF device fingerprinting from IQ samples — RTL-SDR compatible.

Identify and distinguish radio transmitters by their unique physical-layer
characteristics extracted from raw IQ captures.  Works with 8-bit RTL-SDR
files and any numpy complex array.

---

## Features

| Feature | Description |
|---|---|
| **Rise time** | Leading-edge envelope ramp-up speed (µs) |
| **Frequency offset** | Carrier deviation from center (Hz) |
| **Phase noise** | Phase jitter at a given offset (dBc/Hz) |
| **Spectral flatness** | Wiener entropy of the power spectrum [0–1] |
| **Burst shape** | Normalized 16-point envelope silhouette |

Classifiers available: **Random Forest** (default), **SVM**, **k-NN**

---

## Architecture

```
rf-fingerprint/
├── features.py          # IQ loading, burst detection, feature extraction
├── classifier.py        # ML classifier wrapper (train / predict / save / load)
├── visualize.py         # Constellation, spectrum, envelope, burst-shape plots
├── rf_fingerprint.py    # CLI entry point
├── sample_data/
│   ├── generate_samples.py   # Synthetic RTL-SDR capture generator
│   ├── device_A/             # 5 × capture_NN.iq (50 kHz offset, fast rise)
│   └── device_B/             # 5 × capture_NN.iq (200 kHz offset, slow rise)
└── tests/
    ├── test_features.py
    └── test_classifier.py
```

### Data flow

```
.iq file / numpy array
        │
        ▼
   load_iq()           — decode 8-bit RTL-SDR or pass complex array through
        │
        ▼
  detect_burst()       — power-threshold burst detection
        │
        ▼
  extract_features()   — rise_time, freq_offset, phase_noise,
        │                 spectral_flatness, burst_shape
        ▼
 features_to_vector()  — flatten to 1-D float32 feature vector
        │
        ▼
 RFFingerprinter       — sklearn pipeline: train / cross-validate / predict
        │
        ▼
  {"label": "device_A", "confidence": 0.94, "all_probs": {…}}
```

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib
```

> No SDR driver required — works purely on previously recorded IQ files.

Optional, for running tests:

```bash
pip install pytest
```

---

## Quick Start

### 1. Generate synthetic sample data

```bash
python sample_data/generate_samples.py
```

Creates `sample_data/device_A/*.iq` and `sample_data/device_B/*.iq`.

### 2. Inspect features of a capture

```bash
python rf_fingerprint.py features sample_data/device_A/capture_00.iq
```

Output:
```
── RF Fingerprint Features: capture_00.iq ──
  Samples loaded      : 65,536
  Burst range         : [16,384 – 49,152]
  Rise time           : 5.333 µs
  Freq offset         : 49,997.3 Hz
  Phase noise         : -42.8 dBc/Hz
  Spectral flatness   : 0.0012
  Burst shape (16pts) : [0.0, 0.24, 0.97, 1.0, …, 0.23, 0.0]
```

### 3. Train a classifier

```bash
python rf_fingerprint.py train \
    --data-dir sample_data \
    --model-out model.pkl \
    --cv 5
```

The data directory must contain one subdirectory per device label:

```
sample_data/
  device_A/
    capture_00.iq
    capture_01.iq
    ...
  device_B/
    capture_00.iq
    ...
```

### 4. Classify an unknown capture

```bash
python rf_fingerprint.py classify \
    --model model.pkl \
    sample_data/device_A/capture_00.iq
```

Output:
```
── capture_00.iq ──
  Prediction  : device_A
  Confidence  : 94.0%
  All probs   :
    device_A             0.940  ████████████████████
    device_B             0.060  █
```

### 5. Visualize a capture

```bash
python rf_fingerprint.py plot sample_data/device_B/capture_00.iq --label "Device B"
```

Saves a 2×2 PNG with IQ constellation, power spectrum, burst envelope, and
burst shape feature.

### 6. Cross-validate without training

```bash
python rf_fingerprint.py cv --data-dir sample_data --folds 5 --model-type rf
```

---

## Python API

```python
import numpy as np
from features import extract_features, features_to_vector
from classifier import RFFingerprinter
from visualize import plot_all

# --- Feature extraction ---
feats = extract_features("capture.iq", sample_rate=2.4e6)
vec = features_to_vector(feats)
print(f"Rise time: {feats['rise_time_us']:.2f} µs")
print(f"Freq offset: {feats['freq_offset_hz']:.1f} Hz")

# --- Train from numpy arrays ---
fp = RFFingerprinter(model_type="rf", unknown_threshold=0.6, sample_rate=2.4e6)

label_to_arrays = {
    "device_A": [np.load("a1.npy"), np.load("a2.npy")],
    "device_B": [np.load("b1.npy"), np.load("b2.npy")],
}
from classifier import build_dataset_from_arrays
X, y, classes = build_dataset_from_arrays(label_to_arrays)
fp.fit(X, y, classes)

# --- Cross-validate ---
cv = fp.cross_validate(X, y, n_splits=5)
print(f"CV accuracy: {cv['mean_accuracy']:.3f} ± {cv['std_accuracy']:.3f}")

# --- Save / load ---
fp.save("model.pkl")
fp2 = RFFingerprinter.load("model.pkl")

# --- Predict ---
result = fp2.predict("unknown_capture.iq")
print(result["label"], result["confidence"])

# --- Visualize ---
fig = plot_all("capture.iq", sample_rate=2.4e6, label="Device A", save_path="out.png")
```

---

## CLI Reference

```
python rf_fingerprint.py <command> [options]

Commands:
  features  Extract and display RF fingerprint features
  train     Train a classifier on labelled IQ files
  classify  Classify one or more unknown captures
  plot      Generate a summary visualization figure
  cv        Cross-validate classifier on labelled data

Common options:
  --sample-rate HZ     IQ sample rate (default: 2.4e6)
  --threshold DB       Burst detection threshold dB above noise (default: -20)

train options:
  --data-dir DIR       Directory of <label>/*.iq subdirectories (required)
  --model-out FILE     Output model path (default: model.pkl)
  --model-type         rf | svm | knn (default: rf)
  --unknown-threshold  Min confidence to accept prediction (default: 0.6)
  --cv N               Run N-fold CV after training (0 = skip)

classify options:
  --model FILE         Trained model .pkl file (required)
  --json               Also print JSON with raw features

plot options:
  --out FILE           Output image (default: <stem>_rf_fingerprint.png)
  --label TEXT         Transmitter label for figure title
  --show               Display interactive plot window
```

---

## RTL-SDR Capture Tips

Record IQ samples with `rtl_sdr`:

```bash
# 2.4 MSPS, 10 seconds, centered at 433.92 MHz
rtl_sdr -f 433920000 -s 2400000 -n 24000000 capture.iq

# Tune to 915 MHz ISM
rtl_sdr -f 915000000 -s 2400000 -n 24000000 capture.iq
```

Then analyze:

```bash
python rf_fingerprint.py features capture.iq --sample-rate 2.4e6
python rf_fingerprint.py plot capture.iq --sample-rate 2.4e6
```

---

## Notes & Limitations

- **Minimum samples**: burst detection requires at least ~512 samples above threshold. Very short bursts may fall back to full-signal analysis.
- **Phase noise accuracy**: improves significantly with longer captures (>32k samples).
- **Training size**: the included sample data (5 captures × 2 classes) is a demo. For real-world use, collect 20–100+ captures per device with varied conditions.
- **Unknown detection**: controlled by `--unknown-threshold` (default 0.6). Lower = more permissive, higher = stricter.
- **8-bit dynamic range**: RTL-SDR captures have ~48 dB usable dynamic range. Strong interferers may bias feature extraction.

---

## License

MIT
