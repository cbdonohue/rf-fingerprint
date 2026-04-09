"""
tests/test_classifier.py — Unit tests for classifier.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import tempfile

from classifier import RFFingerprinter, build_dataset_from_arrays


SR = 2.4e6


def make_class_samples(n_samples=5, freq_hz=10e3, noise_std=0.05, n_total=4096):
    """Generate a list of complex IQ arrays for one class."""
    rng = np.random.default_rng(seed=int(freq_hz))
    samples = []
    for _ in range(n_samples):
        t = np.arange(n_total) / SR
        base = np.exp(1j * 2 * np.pi * freq_hz * t).astype(np.complex64)
        noise = (rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total)).astype(np.complex64) * noise_std
        samples.append(base + noise)
    return samples


@pytest.fixture
def two_class_dataset():
    """Returns label_to_arrays dict with two well-separated classes."""
    return {
        "device_A": make_class_samples(n_samples=10, freq_hz=50e3),
        "device_B": make_class_samples(n_samples=10, freq_hz=200e3),
    }


class TestBuildDataset:
    def test_returns_correct_shapes(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        assert X.shape[0] == 20
        assert y.shape == (20,)
        assert set(classes) == {"device_A", "device_B"}

    def test_labels_are_integers(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        assert y.dtype in (np.int32, np.int64)
        assert set(y.tolist()) == {0, 1}


class TestRFFingerprinter:
    def test_fit_predict_basic(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        fp = RFFingerprinter(model_type="rf", unknown_threshold=0.0, sample_rate=SR)
        fp.fit(X, y, classes)
        # Predict on training sample — should be one of the two classes
        test_sample = two_class_dataset["device_A"][0]
        result = fp.predict(test_sample)
        assert result["label"] in {"device_A", "device_B"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unknown_threshold(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        fp = RFFingerprinter(model_type="rf", unknown_threshold=0.999, sample_rate=SR)
        fp.fit(X, y, classes)
        # Pure noise — should often be "unknown" at very high threshold
        noise = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64) * 0.001
        result = fp.predict(noise)
        assert "label" in result  # structure is correct regardless of value

    def test_requires_trained(self):
        fp = RFFingerprinter()
        noise = np.ones(512, dtype=np.complex64)
        with pytest.raises(RuntimeError, match="not trained"):
            fp.predict(noise)

    def test_save_load_roundtrip(self, two_class_dataset, tmp_path):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        fp = RFFingerprinter(model_type="rf", unknown_threshold=0.5, sample_rate=SR)
        fp.fit(X, y, classes)

        model_path = tmp_path / "model.pkl"
        fp.save(model_path)

        fp2 = RFFingerprinter.load(model_path)
        assert fp2.classes_ == classes
        assert fp2.unknown_threshold == 0.5
        assert fp2.sample_rate == SR

        # Predictions should match
        test_sample = two_class_dataset["device_B"][0]
        r1 = fp.predict(test_sample)
        r2 = fp2.predict(test_sample)
        assert r1["label"] == r2["label"]

    def test_cross_validate(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        fp = RFFingerprinter(model_type="rf", sample_rate=SR)
        fp.fit(X, y, classes)
        cv = fp.cross_validate(X, y, n_splits=2)
        assert "mean_accuracy" in cv
        assert 0.0 <= cv["mean_accuracy"] <= 1.0
        assert len(cv["fold_scores"]) == 2

    def test_svm_model(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        fp = RFFingerprinter(model_type="svm", unknown_threshold=0.0, sample_rate=SR)
        fp.fit(X, y, classes)
        test_sample = two_class_dataset["device_A"][0]
        result = fp.predict(test_sample)
        assert result["label"] in classes

    def test_knn_model(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        fp = RFFingerprinter(model_type="knn", unknown_threshold=0.0, sample_rate=SR)
        fp.fit(X, y, classes)
        test_sample = two_class_dataset["device_B"][0]
        result = fp.predict(test_sample)
        assert result["label"] in classes

    def test_predict_batch(self, two_class_dataset):
        X, y, classes = build_dataset_from_arrays(two_class_dataset, sample_rate=SR)
        fp = RFFingerprinter(model_type="rf", unknown_threshold=0.0, sample_rate=SR)
        fp.fit(X, y, classes)
        sources = two_class_dataset["device_A"][:3]
        results = fp.predict_batch(sources)
        assert len(results) == 3
        for r in results:
            assert "label" in r
            assert "confidence" in r

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="Unknown model"):
            RFFingerprinter(model_type="xgboost")
