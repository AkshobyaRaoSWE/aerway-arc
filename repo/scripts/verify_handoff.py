"""Verify an extracted Arc review bundle without original workspace paths."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc.dataset import load_manifest
from arc.pipeline import evaluate, load_model, predict, train
from arc.reference import validate_awake_reference
from arc.session import Session


def compare(expected, actual):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys()
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual)
        for left, right in zip(expected, actual):
            compare(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(expected, actual, rtol=1e-10, atol=1e-12)
    else:
        assert expected == actual


def verify(root):
    if not __debug__:
        raise RuntimeError("verification requires assertions enabled; remove -O/PYTHONOPTIMIZE")
    root = Path(root).resolve()
    checksums = json.loads((root / "checksums.json").read_text())
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError("unsafe bundle path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    sample = Session(root / "evidence/sample-session")
    validate_awake_reference(sample, verify_labels=True)
    audit = json.loads((root / "evidence/replay/replay-audit.json").read_text())
    assert audit["source_manifest_sha256"] == sample.content_digest
    counts = {}
    for name, descriptor in sample.streams.items():
        with (root / "evidence/replay" / audit["streams"][name]["file"]).open(newline="") as stream:
            rows = csv.DictReader(stream)
            count = 0
            for first, values, valid in sample.iter_chunks(name):
                for offset, (expected_values, expected_valid) in enumerate(zip(values, valid)):
                    row = next(rows)
                    index = first + offset
                    assert int(row["sample_index"]) == index
                    assert float(row["time_s"]) == descriptor["start_time_s"] + index / descriptor["sample_rate_hz"]
                    assert bool(int(row["valid"])) == bool(expected_valid)
                    np.testing.assert_array_equal([float(row[f"value:{channel}"]) for channel in descriptor["channels"]], expected_values)
                    count += 1
            assert next(rows, None) is None
            assert count == descriptor["n_samples"]
            counts[name] = count
    sessions = load_manifest(root / "evidence/dataset/manifest.json")
    for task in ("awake", "night_audio"):
        base = root / "evidence/baselines" / task
        model = load_model(base / "model.json")
        assert sample.metadata["participant_id"] in model["splits"]["test"]
        stored = json.loads((base / "sample-predictions.json").read_text())
        actual = list(predict(sample, model))
        assert len(stored) == len(actual)
        for left, right in zip(stored, actual):
            assert left["valid"] == right["valid"] and left["reason"] == right["reason"]
            assert left["start_s"] == right["start_s"] and left["end_s"] == right["end_s"]
            if left["valid"]:
                np.testing.assert_allclose(left["probability"], right["probability"], atol=1e-12, rtol=1e-12)
            else:
                assert left["probability"] is right["probability"] is None
        compare(json.loads((base / "test.json").read_text()), evaluate(sessions, model))
        with tempfile.TemporaryDirectory(prefix="arc-refit-") as temporary:
            fitted, validation = train(sessions, model["splits"], task, Path(temporary) / "run", model["window_config"],
                                       seed=model["seed"], allow_synthetic=True,
                                       min_duration_s=model["decision"]["min_duration_s"], gap_merge_s=model["decision"]["gap_merge_s"])
            compare({k: v for k, v in model.items() if k not in ("bundle_digest", "environment")},
                    {k: v for k, v in fitted.items() if k not in ("bundle_digest", "environment")})
            stored_validation = json.loads((base / "validation.json").read_text())
            compare({k: v for k, v in stored_validation.items() if k != "model_digest"},
                    {k: v for k, v in validation.items() if k != "model_digest"})
    original = Session(root / "evidence/crash/interrupted")
    recovered = Session(root / "evidence/crash/recovered")
    assert original.recording_status == "recording" and recovered.recording_status == "recovered"
    assert not original.complete and not recovered.complete
    values, valid = recovered.read("emg", 0, 2)
    np.testing.assert_array_equal(values[:, 0], np.arange(200))
    assert np.flatnonzero(~valid).tolist() == list(range(130, 135))
    assert not (recovered.path / "uncommitted-torn.npy").exists()
    return {"checksummed_files": len(checksums), "csv_exact_roundtrip_sample_counts": counts,
            "both_models_replayed": True, "both_baselines_retrained_within_numeric_tolerance": True, "held_out_reports_reproduced": True,
            "crash_recovery_verified": True, "synthetic_only": True, "clinical_performance_evidence": False}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1]), indent=2))
