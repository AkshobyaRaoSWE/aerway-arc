"""Synthetic software checks only, never evidence of clinical model quality."""

import copy
import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from arc.features import awake_features, log_mel, night_features
from arc.models import fit_awake_threshold, fit_baseline, predict_proba, validate_bundle


def training_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 4))
    return x, (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)


@pytest.mark.parametrize("task", ["awake", "night_audio"])
def test_json_roundtrip_matches_sklearn(task):
    x, y = training_data()
    names = [f"f{i}" for i in range(x.shape[1])]
    bundle = fit_baseline(x, y, task, names, seed=42)
    bundle = json.loads(json.dumps(bundle, allow_nan=False))
    if task == "awake":
        reference = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                           class_weight="balanced", random_state=42, n_jobs=1)
        reference.fit(x, y)
        expected = reference.predict_proba(x)[:, 1]
    else:
        scaler = StandardScaler().fit(x)
        reference = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        reference.fit(scaler.transform(x), y)
        expected = reference.predict_proba(scaler.transform(x))[:, 1]
    np.testing.assert_allclose(predict_proba(bundle, x), expected, rtol=1e-13, atol=1e-14)
    assert bundle["probability_calibration"] == "uncalibrated"


def test_forest_float32_threshold_parity():
    x, y = training_data()
    bundle = fit_baseline(x, y, "awake", ["a", "b", "c", "d"])
    reference = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2,
                                       class_weight="balanced", random_state=42, n_jobs=1).fit(x, y)
    threshold = bundle["trees"][0]["threshold"][0]
    feature = bundle["trees"][0]["feature"][0]
    probes = x[:3].copy()
    probes[:, feature] = [threshold, np.nextafter(threshold, -np.inf), np.nextafter(threshold, np.inf)]
    np.testing.assert_allclose(predict_proba(bundle, probes), reference.predict_proba(probes)[:, 1], atol=1e-14)


@pytest.mark.parametrize("damage", ["cycle", "feature", "probability", "fraction", "leaf"])
def test_reject_corrupt_forest(damage):
    x, y = training_data()
    bundle = fit_baseline(x, y, "awake", ["a", "b", "c", "d"])
    tree = bundle["trees"][0]
    if damage == "cycle":
        tree["left"][0] = 0
    elif damage == "feature":
        tree["feature"][0] = 4
    elif damage == "probability":
        tree["probability"][0] = 1.1
    elif damage == "fraction":
        tree["left"][0] = 1.5
    else:
        tree["left"][0] = -1
    with pytest.raises(ValueError):
        predict_proba(bundle, x)


def test_reject_bad_model_inputs():
    x, y = training_data()
    with pytest.raises(ValueError):
        fit_baseline(x, np.zeros(len(x)), "awake", ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        fit_baseline(x, y, "awake", ["a", "a", "c", "d"])
    bundle = fit_baseline(x, y, "night_audio", ["a", "b", "c", "d"])
    with pytest.raises(ValueError):
        predict_proba(bundle, x[:, :3])
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        predict_proba(bundle, bad)
    bundle["scale"][0] = 0
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_awake_features_causal_dc_invariant_and_calibrated():
    fs = 1600
    t = np.arange(800) / fs
    signal = np.sin(2 * np.pi * 100 * t)[:, None]
    calibration = {"id": "pretrial", "source": "independent calibration", "rms_reference": [2.]}
    values, names = awake_features(signal, fs, calibration)
    shifted, shifted_names = awake_features(signal + 100, fs, calibration)
    np.testing.assert_allclose(values, shifted, atol=1e-11)
    assert names == shifted_names
    assert values[names.index("ch0.rms")] == pytest.approx(1 / np.sqrt(2))
    assert values[names.index("ch0.relative_rms")] == pytest.approx(1 / (2 * np.sqrt(2)))
    longer = np.r_[signal, np.ones((100, 1)) * 1000]
    before, _ = awake_features(longer[:800], fs, calibration)
    np.testing.assert_array_equal(before, values)


def test_awake_features_motion_contract_and_invalid_calibration():
    x = np.ones((800, 2))
    calibration = {"id": "c", "source": "pretrial", "rms_reference": [1., 1.]}
    values, names = awake_features(x, 1600, calibration, np.ones((50, 6)))
    assert len(values) == 12
    assert names[-2:] == ["motion.accel_rms", "motion.gyro_rms"]
    np.testing.assert_allclose(values[-2:], np.sqrt(3))
    with pytest.raises(ValueError):
        awake_features(x, 1600, calibration, np.ones((50, 3)))
    calibration["rms_reference"][0] = 0
    with pytest.raises(ValueError):
        awake_features(x, 1600, calibration)


def test_night_frontend_shapes_and_silence_finite():
    x = np.zeros((32000, 1))
    mel = log_mel(x)
    assert mel.shape == (197, 64)
    values, names = night_features(x, 16000)
    assert len(values) == len(names) == 28
    assert np.isfinite(values).all()
    # Silence features can be computed; the upstream quality gate must still reject detached audio.
    assert values[1] == 0


def test_night_frontend_rejects_nonmono_and_nan():
    with pytest.raises(ValueError):
        night_features(np.zeros((1000, 2)), 16000)
    with pytest.raises(ValueError):
        night_features(np.full((1000, 1), np.nan), 16000)
    with pytest.raises(ValueError):
        night_features(np.zeros((100, 1)), 16000)


def test_amplitude_threshold_export_and_direction():
    x = np.array([[0.1, 10], [0.2, 20], [0.8, 30], [1.0, 40]])
    names = ["ch0.relative_rms", "ch0.rms"]
    bundle = fit_awake_threshold(x, np.array([0, 0, 1, 1]), names)
    bundle = json.loads(json.dumps(bundle, allow_nan=False))
    np.testing.assert_array_equal(predict_proba(bundle, x), [0, 0, 1, 1])
    assert bundle["probability_calibration"] == "hard_threshold_not_probability"
    damaged = copy.deepcopy(bundle)
    damaged["indices"] = [1]
    with pytest.raises(ValueError):
        predict_proba(damaged, x)
