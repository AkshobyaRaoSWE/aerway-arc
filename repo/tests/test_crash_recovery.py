"""Crash tests use actual subprocess exits, never claim hardware power-loss proof."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from arc.session import Session, SessionWriter, recover_session, write_session


def metadata():
    return dict(participant_id="synthetic-p", session_id="synthetic-s", placement_id="synthetic-placement",
                source="crash-software-test", source_version="1", permission="generated fixture",
                synthetic=True, acquisition_config={"clock": "synthetic sample counter"}, label_source="no clinical labels")


def configs():
    return {"emg": {"sample_rate_hz": 10, "channels": ["left", "right"], "units": ["uV", "uV"], "start_time_s": .2}}


def arrays():
    return np.arange(12, dtype=np.int32).reshape(6, 2), np.array([True, True, False, True, True, True])


def test_incremental_checkpoint_replays_exact_committed_samples(tmp_path):
    writer = SessionWriter(tmp_path / "recording", metadata(), configs())
    assert not Session(writer.path).complete
    assert Session(writer.path).duration_s == 0
    samples, valid = arrays()
    first = writer.append("emg", samples[:3], valid[:3])
    assert first["first_sample"] == 0 and first["n_samples"] == 3
    checkpoint = Session(writer.path)
    assert checkpoint.recording_status == "recording"
    actual, mask = checkpoint.read("emg", .2, .5)
    np.testing.assert_array_equal(actual, samples[:3])
    np.testing.assert_array_equal(mask, valid[:3])
    writer.append("emg", samples[3:], valid[3:])
    writer.finalize([])
    completed = Session(writer.path)
    assert completed.complete and completed.duration_s == .8
    np.testing.assert_array_equal(completed.read("emg", .2, .8)[0], samples)
    with pytest.raises(ValueError, match="finalized"):
        writer.append("emg", samples, valid)


@pytest.mark.parametrize("failure", ["after_ack", "torn_array", "before_checkpoint"])
def test_real_process_crash_preserves_acknowledged_checkpoint(tmp_path, failure):
    destination = tmp_path / "crashed"
    script = '''
import json, os, sys
import numpy as np
import arc.session as module
metadata=json.loads(sys.argv[2]); configs=json.loads(sys.argv[3]); mode=sys.argv[4]
writer=module.SessionWriter(sys.argv[1],metadata,configs)
samples=np.arange(12,dtype=np.int32).reshape(6,2)
valid=np.array([True,True,False,True,True,True])
writer.append("emg",samples[:3],valid[:3])
print("acknowledged=3",flush=True)
if mode=="after_ack": os._exit(73)
if mode=="torn_array":
 def torn(path,array):
  with path.open("xb") as f:
   f.write(b"torn NPY");f.flush();os.fsync(f.fileno())
  os._exit(73)
 module._save_array=torn
else:
 def uncommitted(path,manifest): os._exit(73)
 module._publish_manifest=uncommitted
writer.append("emg",samples[3:],valid[3:])
'''
    result = subprocess.run([sys.executable, "-c", script, str(destination), json.dumps(metadata()),
                             json.dumps(configs()), failure], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 73, result.stderr
    assert "acknowledged=3" in result.stdout
    checkpoint = Session(destination)
    assert not checkpoint.complete and checkpoint.recording_status == "recording"
    assert checkpoint.streams["emg"]["n_samples"] == 3
    samples, valid = arrays()
    actual, mask = checkpoint.read("emg", .2, .8)
    np.testing.assert_array_equal(actual[:3], samples[:3])
    np.testing.assert_array_equal(mask[:3], valid[:3])
    assert np.isnan(actual[3:]).all() and not mask[3:].any()
    before = (destination / "session.json").read_bytes()
    recovered = recover_session(destination, tmp_path / "recovered")
    snapshot = Session(recovered)
    assert not snapshot.complete and snapshot.recording_status == "recovered"
    assert snapshot.manifest["recovered_from_digest"] == checkpoint.content_digest
    assert (destination / "session.json").read_bytes() == before
    assert set(p.name for p in recovered.iterdir()) == {"session.json", "emg-000000000000.npy", "emg-000000000000-valid.npy"}
    np.testing.assert_array_equal(snapshot.read("emg", .2, .5)[0], samples[:3])


def test_failed_later_chunk_never_deletes_acknowledged_data(tmp_path, monkeypatch):
    import arc.session as module
    original = module._save_array
    calls = 0
    def fail(path, value):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("disk full during second validity file")
        return original(path, value)
    monkeypatch.setattr(module, "_save_array", fail)
    samples, valid = arrays()
    stream = dict(configs()["emg"], samples=samples, valid=valid)
    with pytest.raises(OSError, match="disk full"):
        write_session(tmp_path / "session", metadata(), {"emg": stream}, [], chunk_samples=3)
    checkpoint = Session(tmp_path / "session")
    assert checkpoint.streams["emg"]["n_samples"] == 3 and not checkpoint.complete
    np.testing.assert_array_equal(checkpoint.read("emg", .2, .5)[0], samples[:3])


def test_missing_declared_stream_cannot_be_finalized(tmp_path):
    config = configs()
    config["audio"] = dict(config["emg"])
    writer = SessionWriter(tmp_path / "session", metadata(), config)
    writer.append("emg", *arrays())
    with pytest.raises(ValueError, match="every declared stream"):
        writer.finalize()
    assert not Session(writer.path).complete


def test_recovery_refuses_overwrite_and_verifies_corruption(tmp_path):
    writer = SessionWriter(tmp_path / "session", metadata(), configs())
    writer.append("emg", *arrays())
    recover_session(writer.path, tmp_path / "recovered")
    with pytest.raises(FileExistsError):
        recover_session(writer.path, tmp_path / "recovered")
    with (writer.path / "emg-000000000000.npy").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        recover_session(writer.path, tmp_path / "bad-recovery")
    assert not (tmp_path / "bad-recovery").exists()


def test_checkpoint_failure_keeps_previous_manifest_and_disables_writer(tmp_path, monkeypatch):
    import arc.session as module
    writer = SessionWriter(tmp_path / "session", metadata(), configs())
    samples, valid = arrays()
    writer.append("emg", samples[:3], valid[:3])
    before = (writer.path / "session.json").read_bytes()
    def fail(path, manifest):
        raise OSError("checkpoint could not publish")
    monkeypatch.setattr(module, "_publish_manifest", fail)
    with pytest.raises(OSError, match="checkpoint"):
        writer.append("emg", samples[3:], valid[3:])
    assert (writer.path / "session.json").read_bytes() == before
    assert Session(writer.path).streams["emg"]["n_samples"] == 3
    with pytest.raises(ValueError, match="failed"):
        writer.finalize()


def test_legacy_manifest_without_status_remains_readable(tmp_path):
    samples, valid = arrays()
    path = write_session(tmp_path / "legacy", metadata(), {"emg": dict(configs()["emg"], samples=samples, valid=valid)}, [])
    filename = path / "session.json"
    manifest = json.loads(filename.read_text())
    del manifest["recording_status"], manifest["declared_streams"]
    filename.write_text(json.dumps(manifest))
    assert Session(path).complete
