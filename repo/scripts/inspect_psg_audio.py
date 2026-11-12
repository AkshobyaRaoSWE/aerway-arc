"""Inspect bounded EDF records plus RML. Never approves alignment or trains a model."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

RESPIRATORY = {"ObstructiveApnea", "CentralApnea", "MixedApnea", "Hypopnea"}


def edf_header(raw):
    if len(raw) < 256 or raw[:8].strip() != b"0" or b"EDF+D" in raw[192:236]:
        raise ValueError("Expected EDF header")
    channels = int(raw[252:256])
    length = int(raw[184:192])
    if not 1 <= channels <= 256 or length != 256 * (channels + 1) or len(raw) < length:
        raise ValueError("Invalid or incomplete EDF header")
    result = {"header_bytes": length, "record_count": int(raw[236:244]),
              "record_duration_s": float(raw[244:252]),
              "start": datetime.strptime(raw[168:184].decode(), "%d.%m.%y%H.%M.%S")}
    offset = 256
    for field, width in [("labels", 16), ("transducers", 80), ("units", 8), ("physical_min", 8),
                         ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
                         ("prefilter", 80), ("samples", 8), ("reserved", 32)]:
        result[field] = [raw[offset + i * width:offset + (i + 1) * width].decode("ascii").strip()
                         for i in range(channels)]
        offset += channels * width
    result["samples"] = [int(x) for x in result["samples"]]
    if (not np.isfinite(result["record_duration_s"]) or result["record_duration_s"] <= 0
            or result["record_count"] <= 0 or min(result["samples"]) <= 0):
        raise ValueError("Invalid EDF record geometry")
    return result


def inspect(header_path, records_path, rml_path, start_record, record_count, source_url):
    if Path(header_path).stat().st_size > 1_000_000:
        raise ValueError("Header input exceeds inspection size limit")
    header_bytes = Path(header_path).read_bytes()
    h = edf_header(header_bytes)
    if start_record < 0 or record_count <= 0 or start_record + record_count > h["record_count"]:
        raise ValueError("Requested record interval outside EDF")
    samples_per_record = sum(h["samples"])
    expected = record_count * samples_per_record * 2
    if expected > 256_000_000:
        raise ValueError("Record range exceeds inspection size limit")
    if Path(records_path).stat().st_size != expected:
        raise ValueError("Incomplete or oversized EDF byte range")
    data = Path(records_path).read_bytes()
    records = np.frombuffer(data, dtype="<i2").reshape(record_count, samples_per_record)
    if Path(rml_path).stat().st_size > 16_000_000:
        raise ValueError("RML exceeds inspection size limit")
    xml = Path(rml_path).read_bytes()
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ValueError("Unsupported RML XML")
    root = ET.fromstring(xml)
    ns = {"p": "http://www.respironics.com/PatientStudy.xsd"}
    sessions = root.findall("p:Acquisition/p:Sessions/p:Session", ns)
    if len(sessions) != 1:
        raise ValueError("Expected exactly one RML acquisition session")
    session = sessions[0]
    origin = datetime.fromisoformat(session.findtext("p:RecordingStart", namespaces=ns))
    duration = float(session.findtext("p:Duration", namespaces=ns))
    segment_offset = (h["start"] - origin).total_seconds()
    start = segment_offset + start_record * h["record_duration_s"]
    end = start + record_count * h["record_duration_s"]
    if not np.isfinite(duration) or start < 0 or end > duration:
        raise ValueError("EDF timestamps fall outside RML acquisition")
    segments = session.findall("p:Segments/p:Segment", ns)
    if any(not np.isfinite(float(s.findtext("p:Duration", namespaces=ns)))
           or float(s.findtext("p:Duration", namespaces=ns)) <= 0 for s in segments):
        raise ValueError("Invalid RML segment duration")
    if not any(datetime.fromisoformat(s.findtext("p:StartTime", namespaces=ns)) == h["start"]
               and float(s.findtext("p:Duration", namespaces=ns)) >= (start_record + record_count) * h["record_duration_s"]
               for s in segments):
        raise ValueError("EDF segment start/duration does not match RML segment")
    events = []
    for event in root.findall("p:ScoringData/p:Events/p:Event", ns):
        if event.get("Family") != "Respiratory" or event.get("Type") not in RESPIRATORY:
            continue
        onset, length = float(event.attrib["Start"]), float(event.attrib["Duration"])
        if not np.isfinite([onset, length]).all() or onset < 0 or length <= 0 or onset + length > duration:
            raise ValueError("Invalid respiratory event timing")
        if onset < end and onset + length > start:
            events.append({"type": event.attrib["Type"], "start_s": onset, "end_s": onset + length})
    offsets = np.r_[0, np.cumsum(h["samples"])]
    audio = {}
    for name, accepted in [("ambient", {"Mic", "Microphone"}), ("tracheal", {"Tracheal"})]:
        matches = [i for i, label in enumerate(h["labels"]) if label in accepted]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous audio channel: " + name)
        i = matches[0]
        rate = h["samples"][i] / h["record_duration_s"]
        if rate != 48000:
            raise ValueError("Unexpected PSG-Audio sample rate")
        values = records[:, offsets[i]:offsets[i + 1]].astype(float)
        if not np.any(values):
            raise ValueError("Audio range contains only padding")
        audio[name] = {"label": h["labels"][i], "sample_rate_hz": rate,
                       "units": h["units"][i] or "unspecified",
                       "samples": int(values.size), "rms_digital": float(np.sqrt(np.mean(values ** 2))),
                       "zero_record_count": int(np.all(values == 0, axis=1).sum()),
                       "clipped_sample_count": int(np.sum((values == -32768) | (values == 32767)))}
    sha = lambda b: hashlib.sha256(b).hexdigest()
    return {"schema_version": 1, "source_url": source_url,
            "provenance": {"header_sha256": sha(header_bytes[:h["header_bytes"]]),
                           "records_sha256": sha(data), "rml_sha256": sha(xml),
                           "byte_range": [h["header_bytes"] + start_record * samples_per_record * 2,
                                          h["header_bytes"] + (start_record + record_count) * samples_per_record * 2 - 1]},
            "edf_start": h["start"].isoformat(), "rml_recording_start": origin.isoformat(),
            "interval_s": [start, end], "record_count": record_count, "audio": audio,
            "overlapping_respiratory_events": events, "structural_ingestion": "PASS",
            "signal_alignment": "UNVALIDATED", "validated_alignment_error_s": None,
            "training_allowed": False, "accuracy_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["header", "records", "rml", "source-url", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--start-record", required=True, type=int)
    parser.add_argument("--record-count", required=True, type=int)
    args = parser.parse_args()
    result = inspect(args.header, args.records, args.rml, args.start_record, args.record_count, args.source_url)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["structural_ingestion", "signal_alignment", "training_allowed"]}))


if __name__ == "__main__":
    main()
