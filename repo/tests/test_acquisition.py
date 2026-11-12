import copy
import json
import uuid

import numpy as np
import pytest

from arc.acquisition import BenchRecorder, FrameDecoder, encode_frame, validate_contract
from arc.session import Session

BOOT, DEVICE_SESSION = str(uuid.UUID(int=1)), str(uuid.UUID(int=2))


def contract(confirmed=False):
    return {"schema_version": 1, "status": "confirmed" if confirmed else "draft",
            "hardware_confirmed_by": "software fixture only", "receiver_confirmed_by": "software fixture only",
            "agreement_record": "test fixture, no hardware agreement", "service_uuid": str(uuid.UUID(int=3)),
            "notify_uuid": str(uuid.UUID(int=4)), "configuration": {"wire_version": 1, "streams": [
                {"id": 1, "name": "emg", "sample_rate_hz": 100, "start_time_s": 0,
                 "dtype_code": 2, "channels": ["a", "b"], "units": ["uV", "uV"],
                 "scale": [.5, .25], "offset": [1., -1.], "max_gap_s": 2,
                 "clock": {"tick_hz": 1000, "first_tick": 0, "first_sample": 0,
                           "mapping_evidence": "synthetic common clock fixture"},
                 "signal_chain": {"source": "synthetic counts; no calibrated hardware"}}]}}


def metadata(real=False):
    return dict(participant_id="bench", session_id="fixture", placement_id="disconnected",
                source="transport_test", source_version="1", permission="test-only", synthetic=not real,
                acquisition_config={}, label_source="none", bench_only=True, body_electrodes_disconnected=True)


def frame(c, seq=0, first=0, ticks=None, samples=None, flags=None, boot=BOOT):
    return encode_frame(c, 1, seq, first, first * 10 if ticks is None else ticks,
                        np.array([[10, -20], [20, -40]]) if samples is None else samples, flags,
                        boot_id=boot, session_id=DEVICE_SESSION)


def test_signed_24bit_fragmented_and_coalesced():
    c = contract()
    values = np.array([[-8388608, 8388607], [-1, 0]])
    packet = frame(c, samples=values)
    decoder = FrameDecoder()
    assert not decoder.feed(packet[:7])
    decoded = decoder.feed(packet[7:] + frame(c, seq=1, first=2))
    np.testing.assert_array_equal(decoded[0]["samples"], values)
    assert len(decoded) == 2 and not decoder.pending


def test_gap_flags_duplicate_and_raw_roundtrip(tmp_path):
    c = contract(); rec = BenchRecorder(tmp_path / "saved", metadata(), c)
    packet = frame(c)
    rec.feed(packet); rec.feed(packet)
    rec.feed(frame(c, seq=2, first=4, flags=np.array([1, 2])))
    session = Session(rec.stop())
    values, valid = session.read("emg", 0, .06)
    np.testing.assert_allclose(values[:2], [[6, -6], [11, -11]])
    assert np.isnan(values[2:4]).all()
    assert valid.tolist() == [True, True, False, False, False, False]
    raw = np.concatenate([part[1] for part in session.iter_chunks("emg_raw")])
    assert raw.dtype == np.int32 and raw[0].tolist() == [10, -20]
    events = [json.loads(line) for line in (session.path / "acquisition.jsonl").read_text().splitlines()]
    gap = next(e for e in events if e["event"] == "sample_gap")
    assert gap["first_sample"] == 2 and gap["end_sample"] == 4


@pytest.mark.parametrize("fault", ["crc", "boot", "config", "clock", "rollback", "gap"])
def test_faults_fail_closed_keep_committed_samples(tmp_path, fault):
    c = contract(); rec = BenchRecorder(tmp_path / fault, metadata(), c)
    rec.feed(frame(c))
    other = copy.deepcopy(c)
    if fault == "config": other["configuration"]["streams"][0]["scale"][0] = 1
    bad = frame(other, seq=1, first=2, ticks=72 if fault == "clock" else None,
                boot=str(uuid.UUID(int=9)) if fault == "boot" else BOOT)
    if fault == "crc": bad = bad[:-1] + bytes([bad[-1] ^ 1])
    if fault == "rollback": bad = frame(c, samples=np.array([[2, 3], [4, 5]]))
    if fault == "gap": bad = frame(c, seq=1, first=4)
    with pytest.raises(ValueError): rec.feed(bad)
    session = Session(tmp_path / fault)
    assert not session.complete
    assert session.read("emg", 0, .02)[1].all()


def test_truncated_frame_and_hardware_gate(tmp_path):
    with pytest.raises(ValueError, match="unconfirmed"):
        BenchRecorder(tmp_path / "real", metadata(True), contract())
    m = metadata(); m["body_electrodes_disconnected"] = False
    with pytest.raises(ValueError, match="disconnected"):
        BenchRecorder(tmp_path / "unsafe", m, contract())
    rec = BenchRecorder(tmp_path / "truncated", metadata(), contract())
    rec.feed(frame(contract())); rec.feed(b"ARC")
    with pytest.raises(ValueError, match="truncated"): rec.stop()
    assert not Session(tmp_path / "truncated").complete


def test_one_minute_clock_continuity_jitter_fragment_replay(tmp_path):
    c = contract(); rec = BenchRecorder(tmp_path / "minute", metadata(), c)
    samples = np.arange(200).reshape(100, 2)
    for seq in range(60):
        packet = frame(c, seq, seq * 100, seq * 1000 + seq % 11, samples=samples)
        rec.feed(packet[:13]); rec.feed(packet[13:])
    session = Session(rec.stop())
    assert session.duration_s == 60
    assert session.read("emg", 0, 60)[1].all()
    assert rec.maximum_clock_residual_s <= .011
    assert session.metadata["acquisition_config"]["clock_alignment_status"].startswith("unverified")
    replay = BenchRecorder(tmp_path / "replay", metadata(), c)
    raw = (session.path / "transport.bin").read_bytes()
    for start in range(0, len(raw), 71): replay.feed(raw[start:start + 71])
    replayed = Session(replay.stop())
    np.testing.assert_array_equal(session.read("emg", 0, 60)[0], replayed.read("emg", 0, 60)[0])


def test_huge_counter_gap_bounded_even_with_bad_tolerance(tmp_path):
    c = contract(); c["configuration"]["streams"][0]["max_gap_s"] = 1e20
    rec = BenchRecorder(tmp_path / "huge", metadata(), c)
    with pytest.raises(ValueError, match="bound"):
        rec.feed(frame(c, seq=1, first=1_000_000))
    assert not Session(tmp_path / "huge").complete


def test_boolean_wire_version_rejected():
    c = contract(); c["configuration"]["wire_version"] = True
    with pytest.raises(ValueError, match="wire_version"):
        validate_contract(c)
