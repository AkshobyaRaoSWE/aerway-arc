"""Release-gate tests use generated arrays and fabricated approvals, not people."""
import json

import numpy as np
import pytest

from arc import pipeline
from arc.reference import labels_from_pressure, validate_awake_reference
from arc.session import Session, write_session


def protocol():
    return {"schema_version": 1, "status": "approved", "protocol_id": "generated-test-only", "protocol_version": "1",
            "instrument_id": "generated", "instrument_model": "not a real instrument", "placement": "test fixture",
            "calibration_id": "generated", "calibration_record": "generated", "source_record": "generated",
            "approved_by": "fabricated fixture, not clinical approval", "approval_reference": "test only",
            "stream": "reference_pressure", "pressure_threshold_kpa": 5., "hold_min_s": .2, "interruption_min_s": .2}


def recording(path, person):
    # The false flag exercises real-data gates, never represents real recordings.
    rule = protocol()
    t = np.arange(6400) / 1600
    emg = (np.sin(2 * np.pi * 100 * t) * np.where((t < 1) | (t >= 3), 20., 2.))[:, None]
    pressure_t = np.arange(400) / 100
    pressure = np.where((pressure_t < 1) | (pressure_t >= 3), 6., 1.)[:, None]
    mask = np.ones(len(pressure), dtype=bool)
    metadata = {"participant_id": person, "session_id": person, "placement_id": "generated",
                "source": "generated-release-test", "source_version": "1", "permission": "generated test data",
                "synthetic": False, "label_source": "generated pressure", "awake_reference": rule,
                "acquisition_config": {"reference_alignment": {"method": "generated shared clock", "max_error_s": 0.},
                                       "signal_chain": {"emg": {"clip_low": -100., "clip_high": 100.}}}}
    streams = {
        "emg": {"sample_rate_hz": 1600, "channels": ["emg0"], "units": ["uV"], "samples": emg,
                "valid": np.ones(len(emg), dtype=bool),
                "calibration": {"id": "generated", "source": "generated", "rms_reference": [10.], "available_at_s": 0.}},
        "reference_pressure": {"sample_rate_hz": 100, "channels": ["tongue_palate_pressure"], "units": ["kPa"],
                               "samples": pressure, "valid": mask},
    }
    write_session(path, metadata, streams, labels_from_pressure(pressure, mask, 100, 0., rule))
    return Session(path)


@pytest.fixture
def cohort(tmp_path):
    return [recording(tmp_path / name, name) for name in ("train", "validation", "test")], {
        name: [name] for name in ("train", "validation", "test")}


def test_real_awake_training_requires_explicit_reference_protocol(cohort, tmp_path):
    sessions, splits = cohort
    del sessions[0].metadata["awake_reference"]
    with pytest.raises(ValueError, match="protocol"):
        pipeline.train(sessions, splits, "awake", tmp_path / "run")


@pytest.mark.parametrize("status", ["synthetic", "pending"])
def test_real_awake_training_rejects_synthetic_or_pending_approval(cohort, tmp_path, status):
    sessions, splits = cohort
    sessions[0].metadata["awake_reference"]["status"] = status
    with pytest.raises(ValueError, match="status|unapproved"):
        pipeline.train(sessions, splits, "awake", tmp_path / "run")


def test_real_awake_training_rejects_absent_reference_stream(cohort, tmp_path):
    sessions, splits = cohort
    del sessions[0].streams["reference_pressure"]
    with pytest.raises(ValueError, match="separate"):
        pipeline.train(sessions, splits, "awake", tmp_path / "run")


def test_real_awake_training_reproduces_labels_from_pressure(cohort, tmp_path):
    sessions, splits = cohort
    sessions[0].labels[0]["state"] = "negative"
    with pytest.raises(ValueError, match="reproduce"):
        pipeline.train(sessions, splits, "awake", tmp_path / "run")


def test_test_reference_is_read_only_during_held_out_evaluation(cohort, tmp_path, monkeypatch):
    sessions, splits = cohort
    heldout = sessions[2]
    original = heldout.read
    def forbidden(*args, **kwargs):
        raise AssertionError("held-out waveform entered training")
    monkeypatch.setattr(heldout, "read", forbidden)
    model, report = pipeline.train(sessions, splits, "awake", tmp_path / "run")
    assert "reference_pressure" not in model["input_signature"]
    assert all("pressure" not in name for name in model["feature_names"])
    assert model["clinical_validation"] == "none" and not model["live_coaching_enabled"]
    calls = []
    def observed(name, start, end):
        calls.append(name)
        return original(name, start, end)
    monkeypatch.setattr(heldout, "read", observed)
    result = pipeline.evaluate(sessions, model)
    assert calls.count("reference_pressure") == 1
    assert "emg" in calls and result["clinical_validation"] == "none"


def test_incomplete_session_cannot_enter_training(cohort, tmp_path):
    sessions, splits = cohort
    sessions[0].complete = False
    sessions[0].recording_status = "recovered"
    with pytest.raises(ValueError, match="incomplete/recovered"):
        pipeline.train(sessions, splits, "awake", tmp_path / "run")


def test_incomplete_session_cannot_enter_evaluation(cohort, tmp_path):
    sessions, splits = cohort
    model, _ = pipeline.train(sessions, splits, "awake", tmp_path / "run")
    sessions[2].complete = False
    sessions[2].recording_status = "recording"
    with pytest.raises(ValueError, match="incomplete/recovered"):
        pipeline.evaluate(sessions, model)


def test_documented_approval_still_requires_complete_reference_fields(cohort):
    session = cohort[0][0]
    del session.metadata["awake_reference"]["calibration_record"]
    with pytest.raises(ValueError, match="calibration_record"):
        validate_awake_reference(session, verify_labels=True)


def test_awake_cohort_cannot_silently_mix_reference_decision_rules(cohort, tmp_path):
    sessions, splits = cohort
    changed = sessions[1]
    rule = changed.metadata["awake_reference"]
    rule["hold_min_s"] = .5
    values, valid = changed.read("reference_pressure", 0., 4.)
    changed.labels = json.loads(json.dumps(labels_from_pressure(values, valid, 100, 0., rule)))
    with pytest.raises(ValueError, match="protocol|reference.*rule|criterion"):
        pipeline.train(sessions, splits, "awake", tmp_path / "run")
