import hashlib
import json

import numpy as np
import pytest

from arc.session import Session, write_session


def metadata():
    return dict(participant_id="p1", session_id="s1", placement_id="placement1",
                source="synthetic_test", source_version="1", permission="test-only",
                synthetic=True, acquisition_config={"clock": "test-aligned"},
                label_source="synthetic_test")


def stream():
    return dict(sample_rate_hz=10, channels=["a", "b"], units=["uV", "uV"],
                samples=np.arange(20, dtype=np.int32).reshape(10, 2),
                valid=np.ones(10, dtype=bool))


def make_session(tmp_path, **kwargs):
    return write_session(tmp_path / "session", metadata(), {"emg": stream()}, [],
                         chunk_samples=3, **kwargs)


def mutate(path, function):
    filename = path / "session.json"
    manifest = json.loads(filename.read_text())
    function(manifest)
    filename.write_text(json.dumps(manifest))


def test_round_trip_chunk_windows_and_raw_identity(tmp_path):
    path = make_session(tmp_path)
    session = Session(path)
    assert session.duration_s == 1
    np.testing.assert_array_equal(session.read("emg", 0.2, 0.7)[0], stream()["samples"][2:7])
    assert session.read("emg", 0.2, 0.7)[1].all()
    chunks = list(session.iter_chunks("emg"))
    assert [chunk[0] for chunk in chunks] == [0, 3, 6, 9]
    assert chunks[0][1].dtype == np.int32
    assert not chunks[0][1].flags.writeable
    np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), stream()["samples"])


def test_missing_samples_not_fabricated(tmp_path):
    values = stream()
    values["start_time_s"] = 0.2
    values["valid"][4] = False
    path = write_session(tmp_path / "session", metadata(), {"emg": values}, [])
    samples, valid = Session(path).read("emg", 0, 1.4)
    assert samples.shape == (14, 2)
    np.testing.assert_array_equal(valid, [False, False, True, True, True, True,
                                         False, True, True, True, True, True, False, False])
    assert np.isnan(samples[:2]).all() and np.isnan(samples[-2:]).all()


def test_off_grid_window_preserves_original_lattice(tmp_path):
    session = Session(make_session(tmp_path))
    np.testing.assert_array_equal(session.read("emg", 0.21, 0.61)[0], stream()["samples"][3:7])
    assert len(session.read("emg", 0.201, 0.202)[0]) == 0


def test_checksum_tamper(tmp_path):
    path = make_session(tmp_path)
    filename = path / "emg-000000000000.npy"
    with filename.open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        Session(path)


@pytest.mark.parametrize("filename", ["../outside.npy", "/etc/passwd", "nested/data.npy", ".."])
def test_path_traversal_even_without_hash_verification(tmp_path, filename):
    path = make_session(tmp_path)
    mutate(path, lambda m: m["streams"]["emg"]["chunks"][0].update(samples=filename))
    with pytest.raises(ValueError, match="paths"):
        Session(path, verify=False)


def test_symlink_rejected(tmp_path):
    path = make_session(tmp_path)
    filename = path / "emg-000000000000.npy"
    outside = tmp_path / "outside.npy"
    filename.rename(outside)
    filename.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        Session(path)


@pytest.mark.parametrize("key,value", [("participant_id", ""), ("permission", " "),
                                        ("synthetic", 1), ("acquisition_config", None)])
def test_metadata_required(tmp_path, key, value):
    values = metadata()
    values[key] = value
    with pytest.raises(ValueError):
        write_session(tmp_path / "session", values, {"emg": stream()}, [])
    assert not (tmp_path / "session").exists()


@pytest.mark.parametrize("change", [
    {"sample_rate_hz": 0}, {"sample_rate_hz": float("nan")},
    {"start_time_s": -1}, {"units": []}, {"channels": ["same", "same"]},
    {"valid": np.ones(10)}, {"valid": np.ones(9, dtype=bool)},
])
def test_stream_validation(tmp_path, change):
    values = stream()
    values.update(change)
    with pytest.raises(ValueError):
        write_session(tmp_path / "session", metadata(), {"emg": values}, [])


def test_nonfinite_only_allowed_in_invalid_samples(tmp_path):
    values = stream()
    values["samples"] = values["samples"].astype(float)
    values["samples"][3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        write_session(tmp_path / "bad", metadata(), {"emg": values}, [])
    values["valid"][3] = False
    path = write_session(tmp_path / "good", metadata(), {"emg": values}, [])
    assert not Session(path).read("emg", 0.3, 0.4)[1][0]


def test_no_overwrite(tmp_path):
    path = make_session(tmp_path)
    before = (path / "session.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_session(path, metadata(), {"emg": stream()}, [])
    assert (path / "session.json").read_bytes() == before


def test_label_provenance_overlap_and_bounds(tmp_path):
    label = dict(start_s=0, end_s=0.5, state="positive", task="awake", provenance="reference")
    for changes in ({"provenance": ""}, {"end_s": 1.1}, {"state": "hold_now"}):
        with pytest.raises(ValueError):
            write_session(tmp_path / "bad", metadata(), {"emg": stream()}, [label | changes])
    with pytest.raises(ValueError, match="overlap"):
        write_session(tmp_path / "bad", metadata(), {"emg": stream()},
                      [label, label | {"start_s": 0.4, "end_s": 1, "state": "negative"}])
    path = write_session(tmp_path / "good", metadata(), {"emg": stream()},
                         [label, label | {"state": "exclude"}])
    assert len(Session(path).labels) == 2


@pytest.mark.parametrize("change", [{"first_sample": 1}, {"n_samples": 99}])
def test_chunk_coverage_validation(tmp_path, change):
    path = make_session(tmp_path)
    mutate(path, lambda m: m["streams"]["emg"]["chunks"][0].update(change))
    with pytest.raises(ValueError, match="tile"):
        Session(path, verify=False)


def test_unknown_schema_and_nonfinite_json(tmp_path):
    path = make_session(tmp_path)
    mutate(path, lambda m: m.update(schema_version=2))
    with pytest.raises(ValueError, match="schema"):
        Session(path)
    mutate(path, lambda m: m.update(schema_version=1, bogus=float("nan")))
    with pytest.raises(ValueError, match="non-finite"):
        Session(path)


def test_manifest_digest_and_mapping_cache(tmp_path):
    path = make_session(tmp_path)
    session = Session(path)
    assert session.content_digest == hashlib.sha256((path / "session.json").read_bytes()).hexdigest()
    before = session._maps["emg-000000000000.npy"]
    session.read("emg", 0.1, 0.2)
    session.read("emg", 0.1, 0.2)
    assert session._maps["emg-000000000000.npy"] is before
    assert len(session._maps) <= 8


def test_metadata_edit_changes_dataset_identity(tmp_path):
    path = make_session(tmp_path)
    before = Session(path).content_digest
    mutate(path, lambda m: m["metadata"].update(placement_id="reattached"))
    assert Session(path).content_digest != before


def test_failed_first_chunk_preserves_empty_recording_checkpoint(tmp_path, monkeypatch):
    import arc.session as module
    original = module._save_array
    calls = 0

    def fail_after_one(path, array):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected disk failure")
        return original(path, array)

    monkeypatch.setattr(module, "_save_array", fail_after_one)
    with pytest.raises(OSError, match="disk failure"):
        make_session(tmp_path)
    session = Session(tmp_path / "session")
    assert session.recording_status == "recording" and not session.complete
    assert session.duration_s == 0 and session.streams == {}


def test_duplicate_json_fields_rejected(tmp_path):
    path = make_session(tmp_path)
    filename = path / "session.json"
    text = filename.read_text().replace('"synthetic": true', '"synthetic": false, "synthetic": true')
    filename.write_text(text)
    with pytest.raises(ValueError, match="duplicate"):
        Session(path)


def test_boolean_schema_rejected(tmp_path):
    path = make_session(tmp_path)
    mutate(path, lambda m: m.update(schema_version=True))
    with pytest.raises(ValueError, match="schema"):
        Session(path)


def test_object_array_rejected_without_executing_pickle(tmp_path):
    path = make_session(tmp_path)
    filename = path / "emg-000000000000.npy"
    np.save(filename, np.array([[{}, {}], [{}, {}], [{}, {}]], dtype=object))
    with pytest.raises(ValueError):
        Session(path, verify=False)


def test_shape_and_invalid_values_checked_without_hashes(tmp_path):
    path = make_session(tmp_path)
    filename = path / "emg-000000000000.npy"
    np.save(filename, np.zeros((9, 2)))
    with pytest.raises(ValueError, match="shape"):
        Session(path, verify=False)
    np.save(filename, np.full((3, 2), np.nan))
    session = Session(path, verify=False)
    with pytest.raises(ValueError, match="NaN"):
        session.read("emg", 0, 0.1)
