#!/usr/bin/env python3
"""
rf_fingerprint.py — CLI entry point for the RF device fingerprinting toolkit.

Usage
-----
# Extract features from a capture
python rf_fingerprint.py features path/to/capture.iq

# Train a classifier from a directory of labelled captures
python rf_fingerprint.py train --data-dir ./sample_data --model-out model.pkl

# Classify an unknown capture
python rf_fingerprint.py classify --model model.pkl path/to/unknown.iq

# Visualize a capture
python rf_fingerprint.py plot path/to/capture.iq --out figure.png

# Cross-validate training set
python rf_fingerprint.py cv --data-dir ./sample_data

Run `python rf_fingerprint.py <command> --help` for full option lists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _glob_iq_files(directory: Path) -> Dict[str, List[Path]]:
    """
    Scan `directory` for subdirectories, treating each subdir name as a
    device label and collecting all .iq / .bin / .raw files inside it.

    Expected layout::

        data_dir/
          device_A/
            capture1.iq
            capture2.iq
          device_B/
            capture1.iq
    """
    mapping: Dict[str, List[Path]] = {}
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir():
            continue
        files = sorted(
            p for p in subdir.iterdir() if p.suffix.lower() in {".iq", ".bin", ".raw"}
        )
        if files:
            mapping[subdir.name] = files
    return mapping


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------

def cmd_features(args: argparse.Namespace) -> None:
    from features import extract_features

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    feats = extract_features(
        path,
        sample_rate=args.sample_rate,
        burst_threshold_db=args.threshold,
    )
    print(f"\n── RF Fingerprint Features: {path.name} ──")
    print(f"  Samples loaded      : {feats['n_samples']:,}")
    print(f"  Burst range         : [{feats['burst_start']:,} – {feats['burst_end']:,}]")
    print(f"  Rise time           : {feats['rise_time_us']:.3f} µs")
    print(f"  Freq offset         : {feats['freq_offset_hz']:.2f} Hz")
    print(f"  Phase noise         : {feats['phase_noise_dbc']:.2f} dBc/Hz")
    print(f"  Spectral flatness   : {feats['spectral_flatness']:.4f}")
    print(f"  Burst shape (16pts) : {np.round(feats['burst_shape'], 3).tolist()}")

    if args.json:
        out = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in feats.items()}
        print("\n── JSON ──")
        print(json.dumps(out, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    from classifier import RFFingerprinter

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    label_to_files = _glob_iq_files(data_dir)
    if not label_to_files:
        print(
            f"No labelled IQ files found in {data_dir}. "
            "Expected structure: data_dir/<label>/*.iq",
            file=sys.stderr,
        )
        sys.exit(1)

    fp = RFFingerprinter(
        model_type=args.model_type,
        unknown_threshold=args.unknown_threshold,
        sample_rate=args.sample_rate,
    )
    fp, X, y, classes = fp.fit_from_files(
        label_to_files,
        burst_threshold_db=args.threshold,
    )

    if args.cv:
        print(f"\nRunning {args.cv}-fold cross-validation …")
        cv_result = fp.cross_validate(X, y, n_splits=args.cv)
        print(f"  CV accuracy: {cv_result['mean_accuracy']:.3f} ± {cv_result['std_accuracy']:.3f}")
        print(f"  Folds      : {[round(s, 3) for s in cv_result['fold_scores']]}")

    model_path = Path(args.model_out)
    fp.save(model_path)
    print(f"\n✓ Trained on {len(X)} samples across {len(classes)} classes.")
    print(f"✓ Model saved to {model_path}")


def cmd_classify(args: argparse.Namespace) -> None:
    from classifier import RFFingerprinter

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    fp = RFFingerprinter.load(model_path)

    files: List[Path] = [Path(f) for f in args.files]
    missing = [f for f in files if not f.exists()]
    if missing:
        for m in missing:
            print(f"Error: file not found: {m}", file=sys.stderr)
        sys.exit(1)

    for fpath in files:
        result = fp.predict(
            fpath,
            burst_threshold_db=args.threshold,
        )
        print(f"\n── {fpath.name} ──")
        print(f"  Prediction  : {result['label']}")
        print(f"  Confidence  : {result['confidence']:.1%}")
        print("  All probs   :")
        for label, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"    {label:<20s} {prob:.3f}  {bar}")

        if args.json:
            out = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result["features"].items()
            }
            print(json.dumps({"label": result["label"], "confidence": result["confidence"], "features": out}, indent=2))


def cmd_plot(args: argparse.Namespace) -> None:
    from visualize import plot_all

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out) if args.out else Path(path.stem + "_rf_fingerprint.png")
    fig = plot_all(
        path,
        sample_rate=args.sample_rate,
        label=args.label or path.stem,
        burst_threshold_db=args.threshold,
        save_path=out,
        show=args.show,
    )
    print(f"✓ Figure saved to {out}")


def cmd_cv(args: argparse.Namespace) -> None:
    from classifier import RFFingerprinter, build_dataset

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    label_to_files = _glob_iq_files(data_dir)
    if not label_to_files:
        print("No labelled IQ files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Building dataset from {sum(len(v) for v in label_to_files.values())} files …")
    X, y, classes = build_dataset(
        label_to_files,
        sample_rate=args.sample_rate,
        burst_threshold_db=args.threshold,
    )
    print(f"  → {len(X)} samples, {len(classes)} classes: {classes}")

    fp = RFFingerprinter(model_type=args.model_type, sample_rate=args.sample_rate)
    fp.fit(X, y, classes)

    print(f"\nRunning {args.folds}-fold cross-validation …")
    cv_result = fp.cross_validate(X, y, n_splits=args.folds)
    print(f"  Mean accuracy : {cv_result['mean_accuracy']:.3f}")
    print(f"  Std           : {cv_result['std_accuracy']:.3f}")
    print(f"  Fold scores   : {[round(s, 3) for s in cv_result['fold_scores']]}")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--sample-rate",
        type=float,
        default=2.4e6,
        metavar="HZ",
        help="IQ sample rate in Hz (default: 2.4e6)",
    )
    shared.add_argument(
        "--threshold",
        type=float,
        default=-20.0,
        metavar="DB",
        help="Burst detection threshold dB above noise floor (default: -20)",
    )

    parser = argparse.ArgumentParser(
        prog="rf_fingerprint",
        description="RF device fingerprinting from IQ samples (RTL-SDR compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # features
    p_feat = sub.add_parser("features", parents=[shared], help="Extract features from a capture")
    p_feat.add_argument("file", help="Path to IQ file (.iq/.bin/.raw)")
    p_feat.add_argument("--json", action="store_true", help="Also print JSON output")

    # train
    p_train = sub.add_parser("train", parents=[shared], help="Train classifier on labelled data")
    p_train.add_argument("--data-dir", required=True, metavar="DIR",
                         help="Directory of <label>/*.iq subdirectories")
    p_train.add_argument("--model-out", default="model.pkl", metavar="FILE",
                         help="Output model path (default: model.pkl)")
    p_train.add_argument("--model-type", choices=["rf", "svm", "knn"], default="rf",
                         help="Classifier type (default: rf)")
    p_train.add_argument("--unknown-threshold", type=float, default=0.6, metavar="P",
                         help="Min confidence to accept prediction (default: 0.6)")
    p_train.add_argument("--cv", type=int, default=0, metavar="N",
                         help="Run N-fold cross-validation after training (0=skip)")

    # classify
    p_cls = sub.add_parser("classify", parents=[shared], help="Classify unknown IQ captures")
    p_cls.add_argument("--model", required=True, metavar="FILE", help="Trained model .pkl file")
    p_cls.add_argument("files", nargs="+", help="IQ file(s) to classify")
    p_cls.add_argument("--json", action="store_true", help="Print JSON with features")

    # plot
    p_plot = sub.add_parser("plot", parents=[shared], help="Visualize an IQ capture")
    p_plot.add_argument("file", help="Path to IQ file")
    p_plot.add_argument("--out", default=None, metavar="FILE",
                        help="Output image path (default: <stem>_rf_fingerprint.png)")
    p_plot.add_argument("--label", default="", help="Transmitter label for figure title")
    p_plot.add_argument("--show", action="store_true", help="Display interactive plot")

    # cv
    p_cv = sub.add_parser("cv", parents=[shared], help="Cross-validate classifier on labelled data")
    p_cv.add_argument("--data-dir", required=True, metavar="DIR")
    p_cv.add_argument("--model-type", choices=["rf", "svm", "knn"], default="rf")
    p_cv.add_argument("--folds", type=int, default=5, metavar="N",
                      help="Number of CV folds (default: 5)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "features": cmd_features,
        "train": cmd_train,
        "classify": cmd_classify,
        "plot": cmd_plot,
        "cv": cmd_cv,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
