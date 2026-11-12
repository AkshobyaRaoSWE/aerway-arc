"""Leakage and abstention checks; generated signals establish no clinical accuracy."""
import copy

import numpy as np
import pytest

from arc import pipeline
from arc.dataset import DEFAULTS, input_signature, iter_windows, label_for
from arc.evaluation import score_predictions
from arc.session import Session, write_session


def session(path, participant, *, clip=False):
    rate = 1600
    t = np.arange(4 * rate) / rate
    samples = (np.sin(2 * np.pi * 100 * t) * np.where((t < 1) | (t >= 3), 20., 2.))[:, None]
    if clip:
        samples[:] = 100.
    metadata = {"participant_id": participant, "session_id": f"session-{participant}", "placement_id": "placement-1",
                "source": "synthetic-review", "source_version": "1", "permission": "generated test data",
                "synthetic": True, "label_source": "generated truth; not clinical measurements",
                "acquisition_config": {"reference_alignment": {"method": "shared generated clock", "max_error_s": 0.},
                                       "signal_chain": {"emg": {"scaling": "generated uV", "clip_low": -100., "clip_high": 100.}}}}
    streams = {"emg": {"sample_rate_hz": rate, "channels": ["regional-emg"], "units": ["uV"],
                        "samples": samples, "valid": np.ones(len(samples), dtype=bool),
                        "calibration": {"id": "before-trial", "source": "synthetic reference", "rms_reference": [10.],
                                        "available_at_s": 0.}}}
    labels = [{"task": "awake", "state": state, "start_s": start, "end_s": end,
               "provenance": "synthetic labels", "confounds": ["speech"] if state == "exclude" else []}
              for start, end, state in [(0., 1., "positive"), (1., 2., "exclude"), (2., 3., "negative"), (3., 4., "positive")]]
    write_session(path, metadata, streams, labels)
    return Session(path)


@pytest.fixture
def cohort(tmp_path):
    recordings = [session(tmp_path / name, name) for name in ("train", "validation", "test")]
    splits = {name: [name] for name in ("train", "validation", "test")}
    return recordings, splits


def test_test_waveforms_never_read_during_fit_or_threshold_selection(cohort, tmp_path, monkeypatch):
    recordings, splits = cohort
    def forbidden_read(*args, **kwargs):
        raise AssertionError("held-out test waveform read during development")
    monkeypatch.setattr(recordings[2], "read", forbidden_read)
    original_fit = pipeline.fit_baseline
    calls = []
    expected_rows, expected_features, _ = pipeline.collect(recordings, splits["train"], "awake", DEFAULTS["awake"],
                                                         input_signature(recordings[0], "awake", DEFAULTS["awake"]))
    expected_X, expected_y = pipeline._labelled(expected_rows, expected_features)
    def observed_fit(X, y, *args, **kwargs):
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)
        calls.append(len(y))
        return original_fit(X, y, *args, **kwargs)
    monkeypatch.setattr(pipeline, "fit_baseline", observed_fit)
    model, report = pipeline.train(recordings, splits, "awake", tmp_path / "run", allow_synthetic=True)
    assert calls == [len(expected_y)]
    assert model["decision"]["threshold_target"] == "interruption"
    assert report["partition"] == "validation"
    assert "Not an unbiased held-out" in report["warning"]


def test_excluded_and_partial_labels_never_become_negative(cohort):
    recording = cohort[0][0]
    windows = list(iter_windows(recording, "awake"))
    excluded = [row for row, _, _ in windows if row["start_s"] >= 1 and row["end_s"] <= 2]
    assert excluded and all(row["label"] is None and row["confounds"] == ["speech"] for row in excluded)
    assert label_for(recording.labels, "awake", .9, 1.1)[0] is None
    assert label_for([], "awake", 0., .1)[0] is None
    collected, features, _ = pipeline.collect([recording], ["train"], "awake", DEFAULTS["awake"],
                                              input_signature(recording, "awake", DEFAULTS["awake"]))
    X, labels = pipeline._labelled(collected, features)
    assert len(X) == sum(row["valid"] and row["label"] is not None for row in collected)
    assert set(labels.tolist()) == {0, 1}


def test_unseen_channel_order_or_signal_chain_abstains(cohort):
    recording = cohort[0][0]
    expected = copy.deepcopy(input_signature(recording, "awake", DEFAULTS["awake"]))
    recording.metadata["acquisition_config"]["signal_chain"]["emg"]["scaling"] = "different amplifier"
    windows = list(iter_windows(recording, "awake", expected_signature=expected))
    assert windows and all(not row["valid"] and vector is None for row, vector, _ in windows)
    assert {row["reason"] for row, _, _ in windows} == {"input_signature_mismatch"}


def test_changed_dataset_rejected_for_frozen_test_evaluation(cohort, tmp_path):
    recordings, splits = cohort
    model, _ = pipeline.train(recordings, splits, "awake", tmp_path / "run", allow_synthetic=True)
    changed = session(tmp_path / "replacement", "replacement")
    with pytest.raises(ValueError, match="dataset changed"):
        pipeline.evaluate(recordings[:2] + [changed], model)


def test_all_invalid_test_has_undefined_metrics_not_a_perfect_score(cohort, tmp_path):
    recordings, splits = cohort
    model, _ = pipeline.train(recordings, splits, "awake", tmp_path / "run", allow_synthetic=True)
    recordings[2].streams["emg"]["calibration"]["available_at_s"] = 100.
    report = pipeline.evaluate(recordings, model)
    assert report["event_sensitivity"] is None
    assert report["event_precision"] is None
    assert report["event_evaluable_seconds"] == 0
    assert report["usable_recording_fraction"] == 0


def test_censoring_removes_event_spans_from_false_alarm_denominator():
    rows = [{"participant_id": "p", "session_id": "s", "task": "night_audio", "start_s": i, "end_s": i + 1,
             "valid": i != 2, "probability": [.9, .9, None, .9, .1][i], "label": [0, 1, 1, 1, 0][i]}
            for i in range(5)]
    report = score_predictions(rows)
    assert report["reference_events"] == report["censored_reference_events"] == 1
    assert report["event_evaluable_seconds"] == 2
    assert report["valid_labelled_seconds"] == 4
    assert report["excluded_predictions"] == 2
    assert report["event_sensitivity"] is None


def test_emg_clipping_abstains_even_when_upstream_mask_is_true(tmp_path):
    recording = session(tmp_path / "clipped", "clipped", clip=True)
    windows = list(iter_windows(recording, "awake"))
    assert all(not row["valid"] for row, _, _ in windows)
    assert "emg_clipping" in {row["reason"] for row, _, _ in windows}
