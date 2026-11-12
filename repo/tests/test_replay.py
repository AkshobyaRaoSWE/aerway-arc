import csv
import hashlib
import json

import numpy as np
import pytest

from arc.replay import _preview, export_replay, iter_replay, plot_session
from arc.session import Session, write_session


def recording(path):
    metadata = dict(participant_id="fixture-person", session_id="fixture-session", placement_id="fixture-placement",
                    source="synthetic replay tests", source_version="1", permission="generated fixture only",
                    synthetic=True, acquisition_config={"clock": "synthetic"}, label_source="synthetic")
    values = (np.arange(16, dtype=np.int64) + 2 ** 54).reshape(8, 2)
    valid = np.array([True, True, False, False, False, True, True, True])
    streams = {"raw": dict(sample_rate_hz=4, start_time_s=.25, channels=["a", "time_s"], units=["count", "count"],
                           samples=values, valid=valid),
               "reference_pressure": dict(sample_rate_hz=2, start_time_s=0., channels=["tongue_palate"], units=["kPa"],
                                          samples=np.array([[0.], [2.], [2.], [0.], [0.], [0.]]), valid=np.ones(6, dtype=bool))}
    write_session(path, metadata, streams, [], chunk_samples=3)
    return Session(path), values, valid


def test_replay_preserves_timestamps_samples_and_masks_across_chunks(tmp_path):
    session, expected, mask = recording(tmp_path / "session")
    blocks = list(iter_replay(session, "raw", block_s=.5))
    assert all(len(block["valid"]) <= 2 for block in blocks)
    np.testing.assert_array_equal(np.concatenate([b["sample_index"] for b in blocks]), np.arange(8))
    np.testing.assert_array_equal(np.concatenate([b["time_s"] for b in blocks]), .25 + np.arange(8) / 4)
    np.testing.assert_array_equal(np.concatenate([b["samples"] for b in blocks]), expected)
    np.testing.assert_array_equal(np.concatenate([b["valid"] for b in blocks]), mask)


def test_csv_round_trip_is_exact_and_audit_merges_chunk_spanning_gaps(tmp_path):
    session, expected, mask = recording(tmp_path / "session")
    original_digest = session.content_digest
    audit = export_replay(session, tmp_path / "replay", block_s=.5)
    with (tmp_path / "replay" / "raw.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_array_equal([[int(row["value:a"]), int(row["value:time_s"])] for row in rows], expected)
    np.testing.assert_array_equal([int(row["valid"]) for row in rows], mask.astype(int))
    np.testing.assert_array_equal([float(row["time_s"]) for row in rows], .25 + np.arange(8) / 4)
    assert audit["source_manifest_sha256"] == original_digest == Session(session.path).content_digest
    stream = audit["streams"]["raw"]
    assert stream["channels"] == ["a", "time_s"] and stream["units"] == ["count", "count"]
    assert stream["valid_sample_count"] == 5 and stream["invalid_sample_count"] == 3
    assert stream["missing_ranges"] == [
        {"first_sample": None, "end_sample": None, "start_s": 0., "end_s": .25, "kind": "not_captured"},
        {"first_sample": 2, "end_sample": 5, "start_s": .75, "end_s": 1.5, "kind": "invalid_samples"},
        {"first_sample": None, "end_sample": None, "start_s": 2.25, "end_s": 3., "kind": "not_captured"},
    ]
    assert stream["sha256"] == hashlib.sha256((tmp_path / "replay/raw.csv").read_bytes()).hexdigest()
    assert json.loads((tmp_path / "replay/replay-audit.json").read_text()) == audit


def test_existing_exports_are_never_overwritten(tmp_path):
    session, _, _ = recording(tmp_path / "session")
    export_replay(session, tmp_path / "replay")
    before = (tmp_path / "replay/raw.csv").read_bytes()
    with pytest.raises(FileExistsError):
        export_replay(session, tmp_path / "replay")
    assert (tmp_path / "replay/raw.csv").read_bytes() == before


@pytest.mark.parametrize("block_s", [0, -1, float("nan"), True, "1"])
def test_invalid_replay_block_size_rejected_before_output(tmp_path, block_s):
    session, _, _ = recording(tmp_path / "session")
    with pytest.raises(ValueError):
        export_replay(session, tmp_path / "bad", block_s)
    assert not (tmp_path / "bad").exists()


def test_plot_preview_cannot_hide_invalid_spans_when_decimated(tmp_path):
    session, _, _ = recording(tmp_path / "session")
    times, values, ranges = _preview(session, "raw", 3, max_points=2)
    assert any(r["start_s"] == .75 and r["end_s"] == 1.5 for r in ranges)
    assert np.isnan(values[times == .75]).all()
    assert not np.any(np.isfinite(values[(times >= .75) & (times < 1.5)]))


def test_plot_is_real_png_and_refuses_overwrite(tmp_path):
    pytest.importorskip("matplotlib")
    session, _, _ = recording(tmp_path / "session")
    output = tmp_path / "session.png"
    result = plot_session(session, output, max_points=4, max_duration_s=2)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert result["synthetic_only"] is True and result["preview_duration_s"] == 2
    with pytest.raises(FileExistsError):
        plot_session(session, output)
    with pytest.raises(ValueError, match="png"):
        plot_session(session, tmp_path / "wrong.svg")
