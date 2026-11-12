"""Session-to-window boundary. Labels never become model inputs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .features import awake_features, night_features
from .session import Session

DEFAULTS = {
    "awake": {"window_s": 0.5, "hop_s": 0.1, "motion": False},
    "night_audio": {"window_s": 2.0, "hop_s": 0.1, "motion": False},
}


def write_json(path, value):
    """Publish complete JSON without replacing another run's artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".arc-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or not isinstance(manifest.get("sessions"), list) or not manifest["sessions"]):
        raise ValueError("manifest requires schema_version=1 and nonempty sessions list")
    sessions = []
    seen = set()
    for relative in manifest["sessions"]:
        if not isinstance(relative, str) or not relative:
            raise ValueError("session paths must be nonempty strings")
        session = Session((path.parent / relative).resolve())
        key = session.metadata["session_id"]
        if key in seen:
            raise ValueError(f"duplicate session_id: {key}")
        seen.add(key)
        sessions.append(session)
    return sessions


def dataset_identity(sessions):
    # The verified session manifest commits to raw chunks, masks and label provenance.
    return fingerprint(sorted(
        [{"session_id": s.metadata["session_id"], "digest": s.content_digest} for s in sessions],
        key=lambda item: item["session_id"],
    ))


def label_for(labels, task, start, end):
    overlapping = [l for l in labels if l["task"] == task and l["start_s"] < end - 1e-9 and l["end_s"] > start + 1e-9]
    confounds = sorted({c for l in overlapping for c in l.get("confounds", [])})
    if any(l["state"] == "exclude" for l in overlapping):
        return None, confounds
    states = {l["state"] for l in overlapping}
    if len(states) != 1:
        return None, confounds
    cursor = start
    for label in sorted(overlapping, key=lambda l: l["start_s"]):
        if label["start_s"] > cursor + 1e-8:
            return None, confounds
        cursor = max(cursor, label["end_s"])
    if cursor < end - 1e-8:
        return None, confounds
    return int(next(iter(states)) == "positive"), confounds


def scored_for(labels, task, start, end):
    spans = [item for item in labels if item["task"] == task and item["start_s"] < end - 1e-9 and item["end_s"] > start + 1e-9]
    if any(item["state"] == "exclude" for item in spans):
        return False
    cursor = start
    for item in sorted(spans, key=lambda item: item["start_s"]):
        if item["start_s"] > cursor + 1e-8:
            return False
        cursor = max(cursor, item["end_s"])
    return bool(cursor >= end - 1e-8)


def reference_events(sessions, ids, task):
    target = "negative" if task == "awake" else "positive"
    events = []
    for session in sessions:
        if session.metadata["participant_id"] not in ids:
            continue
        merged = []
        for label in sorted(session.labels, key=lambda item: item["start_s"]):
            if label["task"] != task or label["state"] != target:
                continue
            if merged and abs(merged[-1]["end_s"] - label["start_s"]) < 1e-9:
                merged[-1]["end_s"] = label["end_s"]
            else:
                merged.append({"participant_id": session.metadata["participant_id"], "session_id": session.metadata["session_id"],
                               "start_s": label["start_s"], "end_s": label["end_s"]})
        events.extend(merged)
    return events


def is_clipped(samples, signal_chain):
    if "clip_low" not in signal_chain and "clip_high" not in signal_chain:
        return False
    try:
        low, high = (np.asarray(signal_chain[key], dtype=float) for key in ("clip_low", "clip_high"))
        low, high = np.broadcast_to(low, (samples.shape[1],)), np.broadcast_to(high, (samples.shape[1],))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("clipping rails need one finite low/high pair per channel") from exc
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
        raise ValueError("clipping rails must be finite and ordered")
    return bool(np.any((samples <= low) | (samples >= high)))


def validate_config(task, config):
    if task not in DEFAULTS or set(config) != set(DEFAULTS[task]):
        raise ValueError("unknown task or window configuration")
    for key in ("window_s", "hop_s"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["window_s"] < config["hop_s"] or not isinstance(config["motion"], bool):
        raise ValueError("window must cover hop; motion must be boolean")
    if task == "night_audio" and config["motion"]:
        raise ValueError("night baseline only supports audio")


def input_signature(session, task, config):
    names = ["emg"] if task == "awake" else ["audio"]
    if config["motion"]:
        names.append("motion")
    signal_chain = session.metadata["acquisition_config"].get("signal_chain", {})
    if not isinstance(signal_chain, dict):
        raise ValueError("acquisition signal_chain must be an object")
    return {name: {**{key: session.streams[name][key] for key in ("sample_rate_hz", "channels", "units")},
                   "signal_chain": signal_chain.get(name)}
            for name in names if name in session.streams}


def iter_windows(session, task, config=None, expected_signature=None):
    """Yield (metadata row, features or None, names); feature context ends at bin end."""
    config = dict(DEFAULTS[task] if config is None else config)
    validate_config(task, config)
    name = "emg" if task == "awake" else "audio"
    descriptor = session.streams.get(name)
    signature = input_signature(session, task, config)
    mismatch = expected_signature is not None and signature != expected_signature
    stream_start = 0.0 if descriptor is None else descriptor["start_time_s"]
    start, end = 0.0, session.duration_s
    hop, window = config["hop_s"], config["window_s"]
    count = int(math.ceil((end - start) / hop - 1e-9))
    for index in range(count):
        left, right = start + index * hop, min(start + (index + 1) * hop, end)
        label, confounds = label_for(session.labels, task, left, right)
        row = {"participant_id": session.metadata["participant_id"], "session_id": session.metadata["session_id"],
               "placement_id": session.metadata["placement_id"], "label_source": session.metadata["label_source"], "task": task, "start_s": left, "end_s": right,
               "context_start_s": right - window, "label": label, "confounds": confounds,
               "scored": scored_for(session.labels, task, left, right),
               "valid": False, "probability": None, "reason": None, "calibration_id": None}
        reason = None
        if descriptor is None:
            reason = "missing_required_stream"
        elif mismatch:
            reason = "input_signature_mismatch"
        elif right - window < stream_start - 1e-9:
            reason = "insufficient_context"
        if reason:
            row["reason"] = reason
            yield row, None, []
            continue
        context_start = max(stream_start, right - window)
        samples, valid = session.read(name, context_start, right)
        if not len(samples) or not valid.all() or not np.isfinite(samples).all():
            row["reason"] = "invalid_samples"
            yield row, None, []
            continue
        signal_chain = signature[name].get("signal_chain") or {}
        if is_clipped(samples, signal_chain):
            row["reason"] = f"{name}_clipping"
            yield row, None, []
            continue
        if task == "awake":
            calibration = descriptor.get("calibration")
            available = calibration.get("available_at_s") if isinstance(calibration, dict) else None
            if (not calibration or not isinstance(calibration.get("id"), str)
                    or isinstance(available, bool) or not isinstance(available, (int, float))
                    or not math.isfinite(available) or available > context_start + 1e-9):
                row["reason"] = "calibration_unavailable"
                yield row, None, []
                continue
            if descriptor["units"] != ["uV"] * samples.shape[1]:
                row["reason"] = "unsupported_units"
                yield row, None, []
                continue
            row["calibration_id"] = calibration["id"]
            motion = None
            if config["motion"]:
                motion_descriptor = session.streams.get("motion")
                if not motion_descriptor or motion_descriptor["units"] != ["m/s^2"] * 3 + ["rad/s"] * 3:
                    row["reason"] = "missing_or_incompatible_motion"
                    yield row, None, []
                    continue
                motion, motion_valid = session.read("motion", context_start, right)
                if not len(motion) or not motion_valid.all() or not np.isfinite(motion).all():
                    row["reason"] = "invalid_motion"
                    yield row, None, []
                    continue
                if is_clipped(motion, signature["motion"].get("signal_chain") or {}):
                    row["reason"] = "motion_clipping"
                    yield row, None, []
                    continue
            features, names = awake_features(samples, descriptor["sample_rate_hz"], calibration, motion)
        else:
            if descriptor["sample_rate_hz"] != 16000 or descriptor["units"] != ["FS"] or samples.shape[1] != 1:
                row["reason"] = "unsupported_audio_format"
                yield row, None, []
                continue
            if np.any(np.abs(samples) >= 1):
                row["reason"] = "audio_clipping"
                yield row, None, []
                continue
            features, names = night_features(samples, descriptor["sample_rate_hz"])
        row["valid"] = True
        yield row, features, names
