"""Adversarial causality checks using synthetic waveforms, not performance evidence."""
import numpy as np

from arc.dataset import DEFAULTS, fingerprint, input_signature, iter_windows
from arc.models import fit_baseline
from arc.pipeline import predict
from arc.session import Session, write_session


def _awake_session(path, suffix, *, mutate_after=None):
    rate = 1600
    t = np.arange(rate * 4) / rate
    samples = (np.sin(2 * np.pi * 100 * t) * np.where(t < 2, 20., 2.))[:, None]
    if mutate_after is not None:
        samples[t >= mutate_after] *= 27
    metadata = {"participant_id": "synthetic-person", "session_id": suffix,
                "placement_id": "test-only", "source": "synthetic-causality-test",
                "source_version": "1", "permission": "generated test waveform",
                "synthetic": True, "acquisition_config": {"scaling": "generated uV"},
                "label_source": "synthetic amplitudes; not clinical truth"}
    streams = {"emg": {"sample_rate_hz": rate, "channels": ["regional-emg"], "units": ["uV"],
                        "samples": samples, "valid": np.ones(len(samples), dtype=bool),
                        "calibration": {"id": "before-trial", "source": "synthetic reference",
                                        "rms_reference": [10.], "available_at_s": 0}}}
    labels = [{"task": "awake", "state": state, "start_s": left, "end_s": right,
               "provenance": "generated test only"}
              for left, right, state in [(0., 2., "positive"), (2., 4., "negative")]]
    write_session(path, metadata, streams, labels)
    return Session(path)


def test_future_samples_cannot_change_past_awake_predictions(tmp_path):
    original = _awake_session(tmp_path / "original", "original")
    changed = _awake_session(tmp_path / "future-changed", "future-changed", mutate_after=2.)
    windows = list(iter_windows(original, "awake"))
    training = [(row, vector, names) for row, vector, names in windows if row["valid"]]
    model = fit_baseline(np.array([vector for _, vector, _ in training]),
                         np.array([row["label"] for row, _, _ in training]),
                         "awake", training[0][2])
    model.update(synthetic_only=True, model_version="test-only",
                 intended_use="research_only", window_config=dict(DEFAULTS["awake"]),
                 input_signature=input_signature(original, "awake", DEFAULTS["awake"]))
    model["bundle_digest"] = fingerprint(model)
    before = list(predict(original, model, batch_size=7))
    after = list(predict(changed, model, batch_size=13))
    for prior, changed_row in zip(before, after):
        if prior["end_s"] <= 2.:
            assert prior["valid"] == changed_row["valid"]
            assert prior["probability"] == changed_row["probability"]
    assert any(a["probability"] != b["probability"] for a, b in zip(before, after) if a["end_s"] > 2.5)


def test_deployment_calibration_after_window_cannot_retroactively_enable_features(tmp_path):
    session = _awake_session(tmp_path / "late", "late")
    session.streams["emg"]["calibration"]["available_at_s"] = 2.
    rows = list(iter_windows(session, "awake"))
    assert all(not row["valid"] for row, _, _ in rows if row["context_start_s"] < 2. - 1e-9)
    assert any(row["valid"] for row, _, _ in rows if row["context_start_s"] >= 2.)
