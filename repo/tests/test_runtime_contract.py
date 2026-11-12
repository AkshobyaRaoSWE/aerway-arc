"""Adversarial runtime checks. All arrays and provenance are fabricated test fixtures."""
import copy
import json

import numpy as np
import pytest

from arc.cli import main
from arc.dataset import DEFAULTS, fingerprint, iter_windows
from arc.models import fit_awake_threshold
from arc.pipeline import load_model, predict
from arc.session import Session, write_session


def _bundle():
    model = fit_awake_threshold(np.array([[.1], [.9]]), np.array([0, 1]), ["ch0.relative_rms"])
    model.update(model_version="synthetic-runtime-contract", intended_use="research_only",
                 synthetic_only=True, clinical_validation="none", live_coaching_enabled=False,
                 window_config=dict(DEFAULTS["awake"]), dataset_digest="0" * 64,
                 input_signature={"emg": {"sample_rate_hz": 1600, "channels": ["emg0"], "units": ["uV"],
                                           "signal_chain": {"clip_low": -1000, "clip_high": 1000}}},
                 decision={"threshold": .5, "threshold_target": "interruption", "min_duration_s": 0.,
                           "gap_merge_s": 0., "iou_threshold": .3},
                 splits={"train": ["p0"], "validation": ["p1"], "test": ["p2"]},
                 provenance=[{"participant_id": f"p{i}", "session_id": f"test-{i}",
                              "source": "synthetic test", "source_version": "1", "permission": "test only",
                              "label_source": "generated test"} for i in range(3)])
    return model


def _save_model(path, model):
    model = copy.deepcopy(model)
    model.pop("bundle_digest", None)
    model["bundle_digest"] = fingerprint(model)
    path.write_text(json.dumps(model, allow_nan=False))
    return path


def _session(path, streams, *, synthetic=True, signal_chain=None):
    metadata = {"participant_id": "test-person", "session_id": path.name, "placement_id": "generated",
                "source": "fabricated runtime fixture", "source_version": "1", "permission": "test only",
                "label_source": "no clinical labels", "synthetic": synthetic,
                "acquisition_config": {"signal_chain": signal_chain or {}}}
    write_session(path, metadata, streams, [])
    return Session(path)


def _emg(seconds=1., start_time_s=0.):
    t = np.arange(round(1600 * seconds)) / 1600
    return {"sample_rate_hz": 1600, "channels": ["emg0"], "units": ["uV"], "start_time_s": start_time_s,
            "samples": np.sin(2 * np.pi * 100 * t)[:, None], "valid": np.ones(len(t), dtype=bool),
            "calibration": {"id": "test", "source": "generated", "rms_reference": [1.], "available_at_s": 0.}}


@pytest.mark.parametrize("field,value", [
    ("threshold", True), ("threshold", 1.01), ("threshold", "NaN"),
    ("min_duration_s", -1), ("gap_merge_s", "0"), ("iou_threshold", 2),
    ("threshold_target", "maintained_hold"),
])
def test_rehashed_invalid_decision_is_rejected(tmp_path, field, value):
    model = _bundle()
    path = tmp_path / "model.json"
    assert load_model(_save_model(path, model))["decision"]["threshold"] == .5
    model["decision"][field] = value
    with pytest.raises(ValueError):
        load_model(_save_model(path, model))


def test_rehashed_missing_decision_is_rejected(tmp_path):
    model = _bundle()
    del model["decision"]
    with pytest.raises(ValueError, match="decision"):
        load_model(_save_model(tmp_path / "model.json", model))


@pytest.mark.parametrize("rail", [-1., 32767 / 32768])
def test_declared_normalized_int16_saturation_abstains(tmp_path, rail):
    samples = np.full((32000, 1), rail)
    streams = {"audio": {"sample_rate_hz": 16000, "channels": ["mic"], "units": ["FS"],
                         "samples": samples, "valid": np.ones(len(samples), dtype=bool)}}
    session = _session(tmp_path / "audio", streams,
                       signal_chain={"audio": {"clip_low": -1., "clip_high": 32767 / 32768,
                                                "scaling": "int16 divided by 32768"}})
    rows = list(iter_windows(session, "night_audio"))
    assert rows[-1][0]["reason"] == "audio_clipping"
    assert rows[-1][0]["valid"] is False
    assert rows[-1][1] is None


@pytest.mark.parametrize("session_synthetic,model_synthetic", [(True, False), (False, True)])
def test_synthetic_and_real_flags_cannot_cross(tmp_path, session_synthetic, model_synthetic):
    # False is exercised only as a flag here; these are never real recordings.
    session = _session(tmp_path / "mismatch", {"emg": _emg()}, synthetic=session_synthetic)
    model = _bundle()
    model["synthetic_only"] = model_synthetic
    with pytest.raises(ValueError, match="cannot be mixed"):
        list(predict(session, model))


def test_short_required_stream_tail_remains_in_coverage(tmp_path):
    streams = {"emg": _emg(),
               "raw": {"sample_rate_hz": 10, "channels": ["counter"], "units": ["count"],
                       "samples": np.ones((20, 1)), "valid": np.ones(20, dtype=bool)}}
    session = _session(tmp_path / "short-emg", streams)
    rows = list(iter_windows(session, "awake"))
    assert len(rows) == 20
    assert rows[0][0]["start_s"] == 0.
    assert rows[-1][0]["end_s"] == 2.
    assert all(not row["valid"] for row, _, _ in rows if row["end_s"] > 1. + 1e-9)
    assert any(row["valid"] for row, _, _ in rows)


def test_late_required_stream_start_remains_in_coverage(tmp_path):
    session = _session(tmp_path / "late-emg", {"emg": _emg(start_time_s=1.)})
    rows = list(iter_windows(session, "awake"))
    assert len(rows) == 20
    assert rows[0][0]["start_s"] == 0.
    assert all(not row["valid"] for row, _, _ in rows if row["end_s"] < 1.5 - 1e-9)
    assert any(row["valid"] for row, _, _ in rows)


def test_cli_missing_path_returns_two(tmp_path, capsys):
    assert main(["validate-session", str(tmp_path / "missing")]) == 2
    assert "arc:" in capsys.readouterr().err


@pytest.mark.parametrize("command,payload", [("split", []), ("pack", {"streams": []})])
def test_cli_malformed_structure_returns_two(tmp_path, capsys, command, payload):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload))
    arguments = (["split", str(path), "--output", str(tmp_path / "out.json")] if command == "split"
                 else ["pack", str(path), str(tmp_path / "out")])
    assert main(arguments) == 2
    assert "arc:" in capsys.readouterr().err


def test_model_from_different_feature_implementation_version_is_rejected(tmp_path):
    model = _bundle()
    model["software_version"] = "0.0.0"
    with pytest.raises(ValueError, match="software version"):
        load_model(_save_model(tmp_path / "old-model.json", model))


def test_declared_motion_saturation_abstains(tmp_path):
    streams = {"emg": _emg(seconds=2.),
               "motion": {"sample_rate_hz": 100, "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
                          "units": ["m/s^2"] * 3 + ["rad/s"] * 3,
                          "samples": np.ones((200, 6)), "valid": np.ones(200, dtype=bool)}}
    session = _session(tmp_path / "motion-clipping", streams,
                       signal_chain={"motion": {"clip_low": -1., "clip_high": 1.}})
    rows = list(iter_windows(session, "awake", dict(DEFAULTS["awake"], motion=True)))
    assert rows[-1][0]["reason"] == "motion_clipping"
    assert not any(row["valid"] for row, _, _ in rows)
