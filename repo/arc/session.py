"""Checksummed, chunked recordings on an explicitly aligned session clock.

Samples are stored unchanged: units and acquisition_config must explain scaling,
channel identity and original ADC configuration. Store original counts as another
stream when physical-unit conversion was applied upstream. No arrival-time clock
alignment, interpolation, lead-off inference, or packet recovery happens here.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


SCHEMA_VERSION = 1
_REQUIRED_TEXT = (
    "participant_id", "session_id", "placement_id", "source", "source_version",
    "permission", "label_source",
)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    for key in _REQUIRED_TEXT:
        _text(metadata.get(key), key)
    if type(metadata.get("synthetic")) is not bool:
        raise ValueError("synthetic must explicitly be true or false")
    if not isinstance(metadata.get("acquisition_config"), dict):
        raise ValueError("acquisition_config must describe the acquisition setup")


def _descriptor(name, stream):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
        raise ValueError("stream names must be simple identifiers")
    if not isinstance(stream, dict):
        raise ValueError(f"{name}: stream must be an object")
    rate = _number(stream.get("sample_rate_hz"), f"{name}.sample_rate_hz")
    if rate == 0:
        raise ValueError("sample_rate_hz must be positive")
    _number(stream.get("start_time_s", 0), f"{name}.start_time_s")
    channels, units = stream.get("channels"), stream.get("units")
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"{name}: channels must be a nonempty list")
    if not isinstance(units, list) or len(units) != len(channels):
        raise ValueError(f"{name}: one unit is required per channel")
    for channel, unit in zip(channels, units):
        _text(channel, "channel")
        _text(unit, "unit")
    if len(set(channels)) != len(channels):
        raise ValueError(f"{name}: channel names must be unique")
    if "calibration" in stream and not isinstance(stream["calibration"], dict):
        raise ValueError(f"{name}: calibration must be an object")


def _arrays(samples, valid, channels):
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise ValueError("samples must be an N x C numpy array")
    if samples.shape[1] != channels or samples.dtype.kind not in "fiu":
        raise ValueError("samples must contain real numeric values for every channel")
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
        raise ValueError("valid must be an explicit boolean numpy array")
    if valid.shape != (len(samples),):
        raise ValueError("valid must contain one flag per sample")
    if not np.isfinite(samples[valid]).all():
        raise ValueError("valid samples cannot contain NaN or infinity")


def _labels(labels, duration):
    if not isinstance(labels, list):
        raise ValueError("labels must be a list")
    spans = {"awake": [], "night_audio": []}
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("each label must be an object")
        start = _number(item.get("start_s"), "label.start_s")
        end = _number(item.get("end_s"), "label.end_s")
        if end <= start or end > duration + 1e-9:
            raise ValueError("label interval must be nonempty and inside session")
        if item.get("state") not in ("positive", "negative", "exclude"):
            raise ValueError("label state must be positive, negative or exclude")
        if item.get("task") not in spans:
            raise ValueError("label task must be awake or night_audio")
        _text(item.get("provenance"), "label.provenance")
        if "confounds" in item:
            if not isinstance(item["confounds"], list):
                raise ValueError("confounds must be a list")
            for value in item["confounds"]:
                _text(value, "confound")
        if item["state"] != "exclude":
            spans[item["task"]].append((start, end))
    for intervals in spans.values():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ValueError("scored labels cannot overlap within a task")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_array(path, array):
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256(path)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_manifest(path, manifest):
    # Commit only after both data files and their directory entries are durable.
    descriptor, temporary = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=path)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path / "session.json")
        _sync_directory(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SessionWriter:
    """Single-writer recorder with an atomic, readable checkpoint per append.

    An append is committed when it returns. A failed/in-flight append may leave
    unreferenced files; readers ignore them. Reopen for reading or recovery after
    failure, never retry this writer. Clock alignment and explicit gaps belong
    to the acquisition adapter; this class does not implement a BLE transport.
    """

    def __init__(self, path, metadata, stream_configs):
        _metadata(metadata)
        if not isinstance(stream_configs, dict) or not stream_configs:
            raise ValueError("at least one stream configuration is required")
        for name, config in stream_configs.items():
            _descriptor(name, config)
            if any(key in config for key in ("samples", "valid", "n_samples", "chunks")):
                raise ValueError("stream configuration must not contain samples or chunk state")
        self.manifest = json.loads(json.dumps({
            "schema_version": SCHEMA_VERSION, "timebase": "aligned_session_seconds",
            "recording_status": "recording", "metadata": metadata,
            "declared_streams": stream_configs, "streams": {}, "labels": [],
        }, allow_nan=False))
        for config in self.manifest["declared_streams"].values():
            config.setdefault("start_time_s", 0.0)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        _sync_directory(self.path.parent)
        self.failed = False
        _publish_manifest(self.path, self.manifest)

    def append(self, stream, samples, valid):
        if self.failed or self.manifest["recording_status"] != "recording":
            raise ValueError("writer is finalized or failed; recover committed data instead")
        if stream not in self.manifest["declared_streams"]:
            raise ValueError("undeclared stream")
        config = self.manifest["declared_streams"][stream]
        _arrays(samples, valid, len(config["channels"]))
        if not len(samples):
            raise ValueError("chunks cannot be empty")
        candidate = json.loads(json.dumps(self.manifest))
        descriptor = candidate["streams"].setdefault(stream, dict(config, n_samples=0, chunks=[]))
        first = descriptor["n_samples"]
        sample_file, valid_file = f"{stream}-{first:012d}.npy", f"{stream}-{first:012d}-valid.npy"
        try:
            chunk = {"first_sample": first, "n_samples": len(samples),
                     "samples": sample_file, "valid": valid_file,
                     "samples_sha256": _save_array(self.path / sample_file, samples),
                     "valid_sha256": _save_array(self.path / valid_file, valid)}
            _sync_directory(self.path)
            descriptor["n_samples"] += len(samples)
            descriptor["chunks"].append(chunk)
            _publish_manifest(self.path, candidate)
        except BaseException:
            self.failed = True
            raise
        self.manifest = candidate
        return dict(chunk)

    def finalize(self, labels=None):
        if self.failed or self.manifest["recording_status"] != "recording":
            raise ValueError("writer is finalized or failed; recover committed data instead")
        if set(self.manifest["streams"]) != set(self.manifest["declared_streams"]):
            raise ValueError("cannot finalize without samples for every declared stream")
        duration = max(d["start_time_s"] + d["n_samples"] / d["sample_rate_hz"]
                       for d in self.manifest["streams"].values())
        labels = [] if labels is None else labels
        _labels(labels, duration)
        candidate = json.loads(json.dumps(self.manifest, allow_nan=False))
        candidate.update(labels=json.loads(json.dumps(labels, allow_nan=False)), recording_status="complete")
        try:
            _publish_manifest(self.path, candidate)
        except BaseException:
            self.failed = True
            raise
        self.manifest = candidate
        return self.path


def write_session(path, metadata, streams, labels, *, chunk_samples=65536):
    """Pack aligned arrays, retaining every committed chunk if interrupted.

    Preflight failures create nothing. Once recording starts, failures preserve
    the readable checkpoint; uncommitted files are ignored. Never overwrites.
    """
    _integer(chunk_samples, "chunk_samples", 1)
    _metadata(metadata)
    if not isinstance(streams, dict) or not streams:
        raise ValueError("at least one stream is required")
    duration, configs = 0.0, {}
    for name, stream in streams.items():
        _descriptor(name, stream)
        _arrays(stream.get("samples"), stream.get("valid"), len(stream["channels"]))
        if not len(stream["samples"]):
            raise ValueError("streams cannot be empty")
        duration = max(duration, stream.get("start_time_s", 0) + len(stream["samples"]) / stream["sample_rate_hz"])
        configs[name] = {k: v for k, v in stream.items() if k not in ("samples", "valid")}
    _labels(labels, duration)
    json.dumps(labels, allow_nan=False)
    writer = SessionWriter(path, metadata, configs)
    for name, stream in streams.items():
        for first in range(0, len(stream["samples"]), chunk_samples):
            writer.append(name, stream["samples"][first:first + chunk_samples], stream["valid"][first:first + chunk_samples])
    return writer.finalize(labels)


def recover_session(source, destination):
    """Copy only the verified committed checkpoint; never promote it to complete.

    Unreferenced/torn files are excluded. Source remains unchanged. This makes a
    reviewable recovery snapshot, not a resumable live writer.
    """
    session = Session(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    _sync_directory(destination.parent)
    manifest = json.loads(json.dumps(session.manifest, allow_nan=False))
    manifest.update(recording_status="recovered", recovered_from_digest=session.content_digest)
    for descriptor in session.streams.values():
        for chunk in descriptor["chunks"]:
            for kind in ("samples", "valid"):
                with session._file(chunk[kind]).open("rb") as original, (destination / chunk[kind]).open("xb") as copied:
                    shutil.copyfileobj(original, copied)
                    copied.flush()
                    os.fsync(copied.fileno())
                if _sha256(destination / chunk[kind]) != chunk[f"{kind}_sha256"]:
                    raise ValueError("source changed during recovery")
    _sync_directory(destination)
    _publish_manifest(destination, manifest)
    return destination


class Session:
    """Validate a saved session; read bounded windows without loading entire nights."""

    def __init__(self, path, verify=True):
        self.path = Path(path).resolve(strict=True)
        manifest_path = self.path / "session.json"
        if manifest_path.is_symlink():
            raise ValueError("manifest must not be a symlink")
        manifest_bytes = manifest_path.read_bytes()
        self.content_digest = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = json.loads(manifest_bytes, parse_constant=self._bad_constant,
                                   object_pairs_hook=self._unique_fields)
        # Bound open mappings while reusing chunks for overlapping windows.
        self._maps = OrderedDict()
        if (not isinstance(self.manifest, dict) or
                type(self.manifest.get("schema_version")) is not int or
                self.manifest["schema_version"] != SCHEMA_VERSION):
            raise ValueError("unsupported session schema")
        if self.manifest.get("timebase") != "aligned_session_seconds":
            raise ValueError("session clock alignment must be explicit")
        self.recording_status = self.manifest.get("recording_status", "complete")
        if self.recording_status not in ("recording", "complete", "recovered"):
            raise ValueError("invalid recording status")
        self.complete = self.recording_status == "complete"
        declared = self.manifest.get("declared_streams")
        if declared is not None:
            if not isinstance(declared, dict) or not declared:
                raise ValueError("declared_streams must be a nonempty object")
            for name, config in declared.items():
                _descriptor(name, config)
        self.metadata = self.manifest.get("metadata")
        self.streams = self.manifest.get("streams")
        self.labels = self.manifest.get("labels")
        _metadata(self.metadata)
        if not isinstance(self.streams, dict) or (self.complete and not self.streams):
            raise ValueError("complete session requires at least one stream")
        if declared is not None:
            if not set(self.streams).issubset(declared) or (self.complete and set(self.streams) != set(declared)):
                raise ValueError("recorded streams differ from declared stream configuration")
            for name, descriptor in self.streams.items():
                _descriptor(name, descriptor)
                if {k: v for k, v in descriptor.items() if k not in ("chunks", "n_samples")} != declared[name]:
                    raise ValueError("recorded stream configuration changed")
        self.duration_s = 0.0
        for name, descriptor in self.streams.items():
            _descriptor(name, descriptor)
            count = _integer(descriptor.get("n_samples"), "n_samples", 1)
            chunks = descriptor.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("stream chunks must be a nonempty list")
            next_sample = 0
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise ValueError("chunk must be an object")
                first = _integer(chunk.get("first_sample"), "first_sample")
                size = _integer(chunk.get("n_samples"), "chunk.n_samples", 1)
                if first != next_sample or first + size > count:
                    raise ValueError("chunks must tile the stream without gaps or overlap")
                next_sample += size
                for kind in ("samples", "valid"):
                    file_path = self._file(chunk.get(kind))
                    expected = chunk.get(f"{kind}_sha256")
                    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                        raise ValueError("invalid SHA256 digest")
                    if verify and _sha256(file_path) != expected:
                        raise ValueError(f"checksum mismatch: {file_path.name}")
                self._load_chunk(descriptor, chunk, check_values=verify)
            if next_sample != count:
                raise ValueError("chunks do not cover declared sample count")
            self.duration_s = max(self.duration_s, descriptor.get("start_time_s", 0) +
                                  count / descriptor["sample_rate_hz"])
        _labels(self.labels, self.duration_s)

    @staticmethod
    def _unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    @staticmethod
    def _bad_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    def _file(self, name):
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("chunk paths must be direct child filenames")
        candidate = self.path / name
        if candidate.is_symlink() or candidate.resolve(strict=True).parent != self.path:
            raise ValueError("chunk path escapes session directory")
        return candidate

    def _array(self, name):
        if name not in self._maps:
            self._maps[name] = np.load(self._file(name), mmap_mode="r", allow_pickle=False)
            if len(self._maps) > 8:
                self._maps.popitem(last=False)
        self._maps.move_to_end(name)
        return self._maps[name]

    def _load_chunk(self, descriptor, chunk, *, check_values=True):
        samples = self._array(chunk["samples"])
        valid = self._array(chunk["valid"])
        if samples.shape != (chunk["n_samples"], len(descriptor["channels"])):
            raise ValueError("chunk shape differs from manifest")
        if check_values:
            _arrays(samples, valid, len(descriptor["channels"]))
        elif samples.dtype.kind not in "fiu" or valid.shape != (len(samples),) or valid.dtype != np.bool_:
            raise ValueError("invalid stored sample or validity type")
        return samples, valid

    def iter_chunks(self, stream):
        """Yield (first_sample, read-only samples, read-only validity) per chunk."""
        descriptor = self.streams[stream]
        for chunk in descriptor["chunks"]:
            samples, valid = self._load_chunk(descriptor, chunk)
            yield chunk["first_sample"], samples, valid

    def read(self, stream, start_s, end_s):
        """Return samples whose original timestamps lie in [start_s, end_s).

        Missing samples outside stream bounds return NaN with false validity.
        Existing values are not interpolated, normalized or filtered. Internal
        gaps must already have false validity. Floating output accommodates NaN.
        """
        start = _number(start_s, "start_s")
        end = _number(end_s, "end_s")
        if end <= start:
            raise ValueError("window must have positive duration")
        descriptor = self.streams[stream]
        rate, origin = descriptor["sample_rate_hz"], descriptor.get("start_time_s", 0)
        first = math.ceil((start - origin) * rate - 1e-9)
        last = math.ceil((end - origin) * rate - 1e-9)
        samples = np.full((last - first, len(descriptor["channels"])), np.nan)
        valid = np.zeros(last - first, dtype=bool)
        for chunk in descriptor["chunks"]:
            left = max(first, chunk["first_sample"])
            right = min(last, chunk["first_sample"] + chunk["n_samples"])
            if right <= left:
                continue
            chunk_samples, chunk_valid = self._load_chunk(descriptor, chunk, check_values=False)
            selection = slice(left - chunk["first_sample"], right - chunk["first_sample"])
            output = slice(left - first, right - first)
            _arrays(chunk_samples[selection], chunk_valid[selection], len(descriptor["channels"]))
            samples[output] = chunk_samples[selection]
            valid[output] = chunk_valid[selection]
        return samples, valid
