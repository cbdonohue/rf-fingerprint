"""
classifier.py — ML classifier for RF device fingerprinting.

Wraps scikit-learn models with training, evaluation, persistence,
and inference helpers.  Supports multi-class (known transmitters) plus
an optional "unknown" detection mode.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from features import extract_features, features_to_vector


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: Dict[str, object] = {
    "rf": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "svm": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]
    ),
    "knn": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]
    ),
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_dataset(
    label_to_files: Dict[str, List[Union[str, Path]]],
    sample_rate: float = 2.4e6,
    burst_threshold_db: float = -20.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a feature matrix X and label vector y from labelled IQ files.

    Parameters
    ----------
    label_to_files : dict mapping label str → list of IQ file paths
    sample_rate    : Hz
    burst_threshold_db : burst detection threshold

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) with integer class indices
    classes : list of label strings (index → label name)
    """
    classes: List[str] = sorted(label_to_files.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for label, files in label_to_files.items():
        idx = class_to_idx[label]
        for fpath in files:
            try:
                feats = extract_features(
                    fpath,
                    sample_rate=sample_rate,
                    burst_threshold_db=burst_threshold_db,
                )
                vec = features_to_vector(feats)
                X_rows.append(vec)
                y_rows.append(idx)
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] Skipping {fpath}: {exc}")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, classes


def build_dataset_from_arrays(
    label_to_arrays: Dict[str, List[np.ndarray]],
    sample_rate: float = 2.4e6,
    burst_threshold_db: float = -20.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Same as build_dataset but accepts numpy arrays instead of file paths."""
    label_to_files_like: Dict[str, list] = {}
    for label, arrays in label_to_arrays.items():
        label_to_files_like[label] = arrays  # arrays pass through load_iq()

    classes: List[str] = sorted(label_to_files_like.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for label, items in label_to_files_like.items():
        idx = class_to_idx[label]
        for item in items:
            try:
                feats = extract_features(
                    item,
                    sample_rate=sample_rate,
                    burst_threshold_db=burst_threshold_db,
                )
                vec = features_to_vector(feats)
                X_rows.append(vec)
                y_rows.append(idx)
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] Skipping array for label '{label}': {exc}")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, classes


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class RFFingerprinter:
    """
    High-level RF device fingerprinting classifier.

    Wraps a scikit-learn estimator with label management, unknown-device
    detection (via confidence thresholding), and model persistence.

    Parameters
    ----------
    model_type   : one of "rf", "svm", "knn"  (see MODELS dict)
    unknown_threshold : minimum prediction probability to accept a label.
                        Below this, the prediction is "unknown".
                        Set to 0.0 to disable unknown detection.
    sample_rate  : Hz — passed through to feature extractor
    """

    def __init__(
        self,
        model_type: str = "rf",
        unknown_threshold: float = 0.6,
        sample_rate: float = 2.4e6,
    ) -> None:
        if model_type not in MODELS:
            raise ValueError(f"Unknown model '{model_type}'. Choose from: {list(MODELS)}")
        self.model = MODELS[model_type]
        self.model_type = model_type
        self.unknown_threshold = unknown_threshold
        self.sample_rate = sample_rate
        self.classes_: Optional[List[str]] = None
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: List[str],
    ) -> "RFFingerprinter":
        """Fit the underlying model on pre-computed feature matrix X."""
        self.classes_ = classes
        self.model.fit(X, y)
        self._trained = True
        return self

    def fit_from_files(
        self,
        label_to_files: Dict[str, List[Union[str, Path]]],
        burst_threshold_db: float = -20.0,
    ) -> Tuple["RFFingerprinter", np.ndarray, np.ndarray, List[str]]:
        """
        Build dataset from labelled IQ files and train.

        Returns (self, X, y, classes) so callers can evaluate afterwards.
        """
        print(f"Building dataset from {sum(len(v) for v in label_to_files.values())} files …")
        X, y, classes = build_dataset(
            label_to_files,
            sample_rate=self.sample_rate,
            burst_threshold_db=burst_threshold_db,
        )
        print(f"  → {len(X)} samples, {len(classes)} classes: {classes}")
        self.fit(X, y, classes)
        return self, X, y, classes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        """Run stratified k-fold CV and return accuracy stats."""
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        return {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
            "fold_scores": scores.tolist(),
        }

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Evaluate on a held-out set (model must already be trained)."""
        self._require_trained()
        y_pred = self.model.predict(X)
        report = classification_report(
            y,
            y_pred,
            target_names=self.classes_,
            output_dict=True,
        )
        cm = confusion_matrix(y, y_pred).tolist()
        return {"classification_report": report, "confusion_matrix": cm}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        self._require_trained()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError(f"Model '{self.model_type}' does not support predict_proba.")

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        burst_threshold_db: float = -20.0,
    ) -> dict:
        """
        Classify a single IQ source (file path or numpy array).

        Returns a dict with:
            label       – predicted class name (or "unknown")
            confidence  – probability of the predicted class
            all_probs   – dict of class → probability
            features    – raw feature dict from extractor
        """
        self._require_trained()
        feats = extract_features(
            source,
            sample_rate=self.sample_rate,
            burst_threshold_db=burst_threshold_db,
        )
        vec = features_to_vector(feats).reshape(1, -1)

        probs = self.predict_proba(vec)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label = self.classes_[best_idx]

        if self.unknown_threshold > 0 and confidence < self.unknown_threshold:
            label = "unknown"

        return {
            "label": label,
            "confidence": confidence,
            "all_probs": {c: float(p) for c, p in zip(self.classes_, probs)},
            "features": feats,
        }

    def predict_batch(
        self,
        sources: Sequence[Union[str, Path, np.ndarray]],
        burst_threshold_db: float = -20.0,
    ) -> List[dict]:
        """Classify multiple IQ sources; returns list of predict() dicts."""
        return [self.predict(s, burst_threshold_db=burst_threshold_db) for s in sources]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialize model + metadata to a pickle file."""
        path = Path(path)
        payload = {
            "model": self.model,
            "model_type": self.model_type,
            "classes": self.classes_,
            "unknown_threshold": self.unknown_threshold,
            "sample_rate": self.sample_rate,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RFFingerprinter":
        """Load a previously saved RFFingerprinter."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.model = payload["model"]
        obj.model_type = payload["model_type"]
        obj.classes_ = payload["classes"]
        obj.unknown_threshold = payload["unknown_threshold"]
        obj.sample_rate = payload["sample_rate"]
        obj._trained = True
        print(f"Model loaded from {path} (classes: {obj.classes_})")
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_trained(self) -> None:
        if not self._trained or self.classes_ is None:
            raise RuntimeError("Model is not trained. Call fit() or fit_from_files() first.")
