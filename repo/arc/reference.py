"""Pressure-derived research labels; instrument choice and clinical criteria are external.

This contract proves traceability and reproducible conversion, not that an
instrument is calibrated, placement is correct, or an approval is authentic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math

import numpy as np


_TEXT_FIELDS = (
    "protocol_id", "protocol_version", "instrument_id", "instrument_model",
    "placement", "calibration_id", "calibration_record", "source_record",
    "approved_by", "approval_reference",
)


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def validate_protocol(protocol, *, synthetic=None):
    if not isinstance(protocol, dict) or type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 1:
        raise ValueError("awake reference requires protocol schema_version=1")
    if protocol.get("status") not in ("approved", "synthetic"):
        raise ValueError("awake reference protocol is unresolved or unapproved")
    if synthetic is not None and protocol["status"] != ("synthetic" if synthetic else "approved"):
        raise ValueError("awake reference approval status does not match recording synthetic status")
    if protocol.get("stream") != "reference_pressure":
        raise ValueError("awake truth requires the separate reference_pressure stream")
    for key in _TEXT_FIELDS:
        if not isinstance(protocol.get(key), str) or not protocol[key].strip():
            raise ValueError(f"awake reference requires documented {key}")
    for key in ("pressure_threshold_kpa", "hold_min_s", "interruption_min_s"):
        _number(protocol.get(key), key, positive=True)
    try:
        json.dumps(protocol, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference protocol must be finite JSON") from exc
    return protocol


def protocol_digest(protocol):
    validate_protocol(protocol)
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()


def labels_from_pressure(samples, valid, sample_rate_hz, start_time_s, protocol):
    """Apply explicit pressure and duration rules to a preserved reference stream.

    Above-threshold runs meeting hold_min_s are positive; below-threshold runs
    meeting interruption_min_s are negative. Shorter runs and invalid reference
    samples are excluded, not silently relabelled. Duration-qualified runs are
    labelled from their original onset: this is offline reference annotation,
    not a live cue policy. Caller must restrict acquisition to the defined task
    or add explicit exclusion annotations for non-task periods.
    """
    validate_protocol(protocol)
    rate = _number(sample_rate_hz, "sample_rate_hz", positive=True)
    origin = _number(start_time_s, "start_time_s")
    samples, valid = np.asarray(samples), np.asarray(valid)
    if samples.ndim != 2 or samples.shape[1] != 1 or samples.dtype.kind not in "fiu":
        raise ValueError("pressure samples must be numeric N x 1 in kPa")
    if valid.dtype != np.bool_ or valid.shape != (len(samples),):
        raise ValueError("reference validity must be one boolean per sample")
    if not np.isfinite(samples[valid]).all():
        raise ValueError("valid pressure samples must be finite")
    if not len(samples):
        raise ValueError("reference stream cannot be empty")
    states = np.full(len(samples), -1, dtype=np.int8)
    states[valid] = (samples[valid, 0] >= protocol["pressure_threshold_kpa"]).astype(np.int8)
    boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(samples)]
    labels = []
    digest = protocol_digest(protocol)
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        state = states[first]
        duration = (last - first) / rate
        minimum = protocol["hold_min_s"] if state == 1 else protocol["interruption_min_s"]
        name = ("positive" if state == 1 else "negative") if state >= 0 and duration + 1e-12 >= minimum else "exclude"
        start, end = float(origin + first / rate), float(origin + last / rate)
        if labels and labels[-1]["state"] == name:
            labels[-1]["end_s"] = end
        else:
            labels.append({"task": "awake", "start_s": start, "end_s": end, "state": name,
                           "provenance": f"pressure-reference:{protocol['protocol_id']}:{protocol['protocol_version']}:{digest}",
                           "reference_protocol_digest": digest})
    return labels


def validate_awake_reference(session, verify_labels=False):
    """Validate provenance; optionally reproduce labels without consuming EMG.

    Metadata-only validation reads no reference waveform, so held-out participants
    can pass schema checks without being used during model development. Label
    verification permits explicit exclusion overlays, but rejects missing or
    contradictory scored truth. Missing hardware/approval cannot pass this gate.
    """
    protocol = validate_protocol(session.metadata.get("awake_reference"), synthetic=session.metadata["synthetic"])
    stream = session.streams.get("reference_pressure")
    if (not isinstance(stream, dict) or stream.get("units") != ["kPa"]
            or stream.get("channels") != ["tongue_palate_pressure"]):
        raise ValueError("awake labels require separate tongue_palate_pressure channel in kPa")
    alignment = session.metadata.get("acquisition_config", {}).get("reference_alignment")
    if not isinstance(alignment, dict) or not isinstance(alignment.get("method"), str) or not alignment["method"].strip():
        raise ValueError("pressure reference needs documented clock alignment")
    _number(alignment.get("max_error_s"), "reference_alignment.max_error_s")
    digest = protocol_digest(protocol)
    if not verify_labels:
        return digest
    origin = stream.get("start_time_s", 0.)
    end = origin + stream["n_samples"] / stream["sample_rate_hz"]
    samples, valid = session.read("reference_pressure", origin, end)
    expected = labels_from_pressure(samples, valid, stream["sample_rate_hz"], origin, protocol)
    actual = [label for label in session.labels if label["task"] == "awake"]
    for label in actual:
        if label["state"] != "exclude" and (label["start_s"] < origin - 1e-9 or label["end_s"] > end + 1e-9):
            raise ValueError("awake scored label lies outside pressure reference coverage")
        if label["state"] != "exclude" and label.get("reference_protocol_digest") != digest:
            raise ValueError("awake scored label lacks matching pressure protocol digest")
    changes = defaultdict(list)
    for label in actual:
        left, right = max(origin, label["start_s"]), min(end, label["end_s"])
        if left < right:
            changes[left].append((label["state"], 1))
            changes[right].append((label["state"], -1))
    points = sorted({origin, end, *changes, *[label["end_s"] for label in expected]})
    active, reference_index = Counter(), 0
    for left, right in zip(points[:-1], points[1:]):
        for state, delta in changes[left]:
            active[state] += delta
        while expected[reference_index]["end_s"] <= left and reference_index + 1 < len(expected):
            reference_index += 1
        if right - left < 1e-12 or active["exclude"]:
            continue
        expected_state = expected[reference_index]["state"]
        if active[expected_state] != 1 or sum(active.values()) != 1:
            raise ValueError("awake labels do not reproduce the preserved pressure reference")
    return digest
