"""Format checks use generated fixtures; these are not model accuracy tests."""
import numpy as np
import pytest

from scripts.inspect_psg_audio import inspect


@pytest.fixture
def recording(tmp_path):
    def padded(value, width):
        return str(value).encode().ljust(width)
    header = bytearray(b" " * 256)
    for start, width, value in [(0, 8, "0"), (168, 8, "24.04.19"), (176, 8, "22.11.51"),
                                (184, 8, 768), (236, 8, 2), (244, 8, 1), (252, 4, 2)]:
        header[start:start + width] = padded(value, width)
    fields = [(16, ["Mic", "Tracheal"]), (80, ["Microphone", "Contact"]), (8, ["", ""]),
              (8, [-1, -1]), (8, [1, 1]), (8, [-32768, -32768]), (8, [32767, 32767]),
              (80, ["", ""]), (8, [48000, 48000]), (32, ["", ""])]
    for width, values in fields:
        header.extend(b"".join(padded(x, width) for x in values))
    (tmp_path / "header").write_bytes(header)
    (tmp_path / "records").write_bytes(np.ones(96000, dtype="<i2").tobytes())
    (tmp_path / "labels").write_text('''<PatientStudy xmlns="http://www.respironics.com/PatientStudy.xsd">
      <Acquisition><Sessions><Session><RecordingStart>2019-04-24T22:11:51</RecordingStart><Duration>2</Duration>
      <Segments><Segment><StartTime>2019-04-24T22:11:51</StartTime><Duration>2</Duration></Segment></Segments>
      </Session></Sessions></Acquisition><ScoringData><Events>
      <Event Family="Respiratory" Type="Hypopnea" Start="0.5" Duration="1"/>
      </Events></ScoringData></PatientStudy>''')
    return tmp_path


def run(root, start=0):
    return inspect(root / "header", root / "records", root / "labels", start, 1, "fixture")


def test_real_format_is_not_alignment_approval(recording):
    result = run(recording, 1)
    assert result["interval_s"] == [1, 2]
    assert result["provenance"]["byte_range"] == [192768, 384767]
    assert result["overlapping_respiratory_events"] == [{"type": "Hypopnea", "start_s": .5, "end_s": 1.5}]
    assert result["structural_ingestion"] == "PASS"
    assert result["signal_alignment"] == "UNVALIDATED"
    assert result["training_allowed"] is False
    assert result["accuracy_claim"] is False


def test_truncated_range_rejected(recording):
    path = recording / "records"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte range"):
        run(recording)


def test_wrong_segment_timestamp_rejected(recording):
    path = recording / "labels"
    path.write_text(path.read_text().replace("<StartTime>2019-04-24T22:11:51", "<StartTime>2019-04-24T22:11:52"))
    with pytest.raises(ValueError, match="segment"):
        run(recording)


def test_padding_and_invalid_event_rejected(recording):
    path = recording / "records"
    path.write_bytes(bytes(path.stat().st_size))
    with pytest.raises(ValueError, match="padding"):
        run(recording)
    path = recording / "labels"
    path.write_text(path.read_text().replace('Duration="1"', 'Duration="NaN"'))
    with pytest.raises(ValueError, match="event timing"):
        run(recording)


def test_discontinuous_edf_rejected(recording):
    path = recording / "header"
    header = bytearray(path.read_bytes())
    header[192:197] = b"EDF+D"
    path.write_bytes(header)
    with pytest.raises(ValueError, match="EDF header"):
        run(recording)


def test_oversized_xml_rejected_before_parsing(recording):
    with (recording / "labels").open("wb") as handle:
        handle.truncate(16_000_001)
    with pytest.raises(ValueError, match="size limit"):
        run(recording)
