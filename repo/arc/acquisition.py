"""Draft Arc binary bench protocol; firmware agreement is an explicit prerequisite."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import uuid
import zlib

import numpy as np

from .session import SessionWriter

MAGIC = b"ARC1"
HEADER = struct.Struct("<4sIBBHI16s16s32sQQIH")
CRC = struct.Struct("<I")
DTYPES = {1: ("int16", 2), 2: ("int24", 3), 3: ("int32", 4)}
CLIPPED, LEAD_OFF, CONTACT_BAD, DEVICE_GAP = 1, 2, 4, 8
MAX_FRAME_BYTES = 1024 * 1024
MAX_GAP_VALUES = 1_000_000


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _integer(value, name, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def configuration_digest(contract):
    """Firmware hashes only the shared wire/acquisition configuration, not signoff."""
    return hashlib.sha256(json.dumps(contract["configuration"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_contract(contract, require_confirmed=False):
    if not isinstance(contract, dict) or contract.get("schema_version") != 1 or type(contract.get("schema_version")) is not int:
        raise ValueError("contract requires schema_version1")
    if contract.get("status") not in ("draft", "confirmed"):
        raise ValueError("contract status must be draft or confirmed")
    if require_confirmed:
        if contract["status"] != "confirmed":
            raise ValueError("hardware contract remains unconfirmed")
        for field in ("hardware_confirmed_by", "receiver_confirmed_by", "agreement_record"):
            if not isinstance(contract.get(field), str) or not contract[field].strip():
                raise ValueError(f"confirmed contract requires {field}")
        for field in ("service_uuid", "notify_uuid"):
            try:
                uuid.UUID(contract.get(field, ""))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"confirmed contract requires actual {field}") from exc
    configuration = contract.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("wire_version") != 1 or type(configuration.get("wire_version")) is not int:
        raise ValueError("wire_version1 configuration required")
    streams = configuration.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("contract requires stream configurations")
    ids, names = set(), set()
    for stream in streams:
        if not isinstance(stream, dict):
            raise ValueError("stream configuration must be an object")
        identifier = _integer(stream.get("id"), "stream id", 65535)
        name = stream.get("name")
        if identifier in ids or not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError("stream IDs/names must be unique simple identifiers")
        ids.add(identifier); names.add(name)
        raw_name = name + "_raw"
        if raw_name in names or any(s.get("name") == raw_name for s in streams):
            raise ValueError("stream name collides with raw-count companion")
        _number(stream.get("sample_rate_hz"), "sample rate", True)
        _number(stream.get("start_time_s"), "stream origin")
        _integer(stream.get("dtype_code"), "dtype code", 3)
        if stream["dtype_code"] not in DTYPES:
            raise ValueError("only signed little-endian integer sample formats supported")
        channels, units = stream.get("channels"), stream.get("units")
        if not isinstance(channels, list) or not 1 <= len(channels) <= 32 or any(not isinstance(c, str) or not c for c in channels) or len(set(channels)) != len(channels):
            raise ValueError("one to32 unique channel names required")
        if not isinstance(units, list) or len(units) != len(channels) or any(not isinstance(u, str) or not u for u in units):
            raise ValueError("one physical unit per channel required")
        for field in ("scale", "offset"):
            values = np.asarray(stream.get(field), dtype=float)
            if values.shape != (len(channels),) or not np.isfinite(values).all() or (field == "scale" and np.any(values == 0)):
                raise ValueError(f"explicit finite per-channel {field} required")
        clock = stream.get("clock")
        if not isinstance(clock, dict):
            raise ValueError("per-stream device clock mapping required")
        _number(clock.get("tick_hz"), "device tick frequency", True)
        _integer(clock.get("first_tick"), "first tick", 2**64 - 1)
        _integer(clock.get("first_sample"), "first sample", 2**64 - 1)
        if clock["first_sample"] != 0:
            raise ValueError("capture starts at counter0; explicit missing prefixes are supported")
        for field in ("mapping_evidence", "signal_chain"):
            if field == "mapping_evidence":
                if not isinstance(clock.get(field), str) or not clock[field].strip():
                    raise ValueError("clock mapping evidence required")
            elif not isinstance(stream.get(field), dict) or not stream[field]:
                raise ValueError("signal_chain must document routing, bandwidth and scaling")
        _number(stream.get("max_gap_s"), "maximum recoverable gap", True)
    json.dumps(contract, allow_nan=False)
    return contract


def encode_frame(contract, stream_id, sequence, first_sample, device_ticks, samples, flags=None, *, boot_id, session_id):
    """Reference encoder for firmware conformance fixtures, not hardware evidence."""
    validate_contract(contract)
    stream = next((s for s in contract["configuration"]["streams"] if s["id"] == stream_id), None)
    if stream is None:
        raise ValueError("unknown stream")
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != len(stream["channels"]) or samples.dtype.kind not in "iu" or not 1 <= len(samples) <= 65536:
        raise ValueError("integer N x C samples required")
    _integer(sequence, "sequence", 2**32 - 1); _integer(first_sample, "sample counter", 2**64 - 1); _integer(device_ticks, "device ticks", 2**64 - 1)
    width = DTYPES[stream["dtype_code"]][1]
    if np.any(samples < -(2**(width * 8 - 1))) or np.any(samples > 2**(width * 8 - 1) - 1):
        raise ValueError("sample outside declared signed ADC range")
    if width == 3:
        raw = samples.astype(np.int32).view(np.uint32).reshape(-1)
        payload = np.stack((raw & 255, (raw >> 8) & 255, (raw >> 16) & 255), axis=1).astype(np.uint8).tobytes()
    else:
        payload = samples.astype(f"<i{width}").tobytes()
    flags = np.zeros(len(samples), dtype=np.uint8) if flags is None else np.asarray(flags)
    if flags.shape != (len(samples),) or flags.dtype.kind not in "iu" or np.any(flags < 0) or np.any(flags > 15):
        raise ValueError("one known quality-flag byte per sample required")
    size = HEADER.size + len(payload) + len(flags) + CRC.size
    if size > MAX_FRAME_BYTES:
        raise ValueError("frame exceeds size limit")
    header = HEADER.pack(MAGIC, size, 1, stream["dtype_code"], stream_id, sequence, uuid.UUID(boot_id).bytes,
                         uuid.UUID(session_id).bytes, bytes.fromhex(configuration_digest(contract)), first_sample,
                         device_ticks, len(samples), samples.shape[1])
    packet = header + payload + flags.astype(np.uint8).tobytes()
    return packet + CRC.pack(zlib.crc32(packet))


class FrameDecoder:
    """Reassemble a bounded binary byte stream across BLE notification boundaries."""
    def __init__(self):
        self.pending = bytearray()

    def feed(self, data):
        if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_FRAME_BYTES:
            raise ValueError("invalid notification bytes")
        self.pending.extend(data)
        frames = []
        while len(self.pending) >= 8:
            if self.pending[:4] != MAGIC:
                raise ValueError("invalid frame magic; do not guess resynchronization")
            size = struct.unpack_from("<I", self.pending, 4)[0]
            if not HEADER.size + CRC.size < size <= MAX_FRAME_BYTES:
                raise ValueError("invalid frame length")
            if len(self.pending) < size:
                break
            packet = bytes(self.pending[:size]); del self.pending[:size]
            if zlib.crc32(packet[:-4]) != CRC.unpack(packet[-4:])[0]:
                raise ValueError("frame checksum mismatch")
            magic, size, version, dtype, stream, sequence, boot, session, digest, first, ticks, count, channels = HEADER.unpack_from(packet)
            if version != 1 or dtype not in DTYPES or not 1 <= count <= 65536 or not 1 <= channels <= 32:
                raise ValueError("unsupported frame version/shape/type")
            width = DTYPES[dtype][1]
            payload_end = HEADER.size + count * channels * width
            if payload_end + count + CRC.size != size:
                raise ValueError("frame shape does not match byte length")
            raw = packet[HEADER.size:payload_end]
            if width == 3:
                octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
                values = (values ^ 0x800000) - 0x800000
            else:
                values = np.frombuffer(raw, dtype=f"<i{width}").astype(np.int32)
            flags = np.frombuffer(packet[payload_end:-4], dtype=np.uint8)
            if np.any(flags > 15):
                raise ValueError("unknown quality flag")
            frames.append({"stream_id": stream, "dtype_code": dtype, "sequence": sequence, "boot_id": str(uuid.UUID(bytes=boot)),
                           "device_session_id": str(uuid.UUID(bytes=session)), "configuration_digest": digest.hex(),
                           "first_sample": first, "device_ticks": ticks, "samples": values.reshape(count, channels), "flags": flags,
                           "frame_sha256": hashlib.sha256(packet).hexdigest()})
        if len(self.pending) > MAX_FRAME_BYTES:
            raise ValueError("fragment buffer exceeds limit")
        return frames


class BenchRecorder:
    """One device boot/session, immutable config, explicit lost ranges, no coaching."""
    def __init__(self, path, metadata, contract):
        self.contract = json.loads(json.dumps(validate_contract(contract, require_confirmed=not metadata.get("synthetic", False))))
        if metadata.get("bench_only") is not True or metadata.get("body_electrodes_disconnected") is not True:
            raise ValueError("bench capture requires explicit disconnected body electrodes")
        self.streams = {s["id"]: s for s in self.contract["configuration"]["streams"]}
        configs = {}
        metadata = json.loads(json.dumps(metadata))
        acquisition = metadata.setdefault("acquisition_config", {})
        acquisition["packet_contract"] = self.contract
        acquisition["signal_chain"] = {s["name"]: s["signal_chain"] for s in self.streams.values()}
        acquisition["clock_alignment_status"] = "unverified against independent reference"
        acquisition["device_final_counter_acknowledged"] = False
        acquisition["completion_scope"] = "locally received window; device tail completeness unverified"
        for stream in self.streams.values():
            descriptor = {key: stream[key] for key in ("sample_rate_hz", "channels", "units", "start_time_s")}
            if "calibration" in stream:
                descriptor["calibration"] = stream["calibration"]
            configs[stream["name"]] = descriptor
            configs[stream["name"] + "_raw"] = {**descriptor, "units": ["count"] * len(stream["channels"])}
        self.writer = SessionWriter(path, metadata, configs)
        self.decoder = FrameDecoder()
        self.counters = {key: {"next_sample": 0, "next_sequence": 0, "frames": 0, "missing_samples": 0} for key in self.streams}
        self.last_frames = {}
        self.identity = None
        self.failed = False
        self.closed = False
        self.maximum_clock_residual_s = 0.
        self.transport_hash = hashlib.sha256()
        self.transport = (self.writer.path / "transport.bin").open("xb")
        self.log = (self.writer.path / "acquisition.jsonl").open("x")

    def _log(self, event):
        self.log.write(json.dumps(event, allow_nan=False) + "\n"); self.log.flush(); os.fsync(self.log.fileno())

    def abort(self, reason):
        if not self.closed:
            self.failed = True
            try:
                self._log({"event": "capture_failed", "reason": str(reason)})
            finally:
                self.transport.close(); self.log.close(); self.closed = True

    def feed(self, data):
        if self.closed or self.failed:
            raise ValueError("recorder stopped or failed")
        try:
            self.transport.write(data); self.transport.flush(); os.fsync(self.transport.fileno()); self.transport_hash.update(data)
            frames = self.decoder.feed(data)
            for frame in frames:
                self._commit(frame)
            return len(frames)
        except BaseException as exc:
            self.abort(exc)
            raise

    def _commit(self, frame):
        stream = self.streams.get(frame["stream_id"])
        if stream is None or frame["dtype_code"] != stream["dtype_code"] or frame["samples"].shape[1] != len(stream["channels"]):
            raise ValueError("frame channel/type differs from contract")
        if frame["configuration_digest"] != configuration_digest(self.contract):
            raise ValueError("device configuration changed or differs from contract")
        identity = frame["boot_id"], frame["device_session_id"]
        if self.identity is not None and identity != self.identity:
            raise ValueError("device boot/session changed; start a new recording")
        self.identity = identity
        state = self.counters[frame["stream_id"]]
        first, count = frame["first_sample"], len(frame["samples"])
        if first < state["next_sample"] or frame["sequence"] < state["next_sequence"]:
            self._log({"event": "duplicate_or_reordered", "stream": stream["name"], "first_sample": first, "sequence": frame["sequence"]})
            if self.last_frames.get(frame["stream_id"]) == frame["frame_sha256"]:
                return
            raise ValueError("counter rollback or unknown reordered frame; start a new recording")
        gap, lost_frames = first - state["next_sample"], frame["sequence"] - state["next_sequence"]
        if bool(gap) != bool(lost_frames):
            raise ValueError("sequence gap and sample-counter gap disagree")
        if gap > stream["max_gap_s"] * stream["sample_rate_hz"] or gap * len(stream["channels"]) > MAX_GAP_VALUES:
            raise ValueError("gap exceeds configured recovery bound")
        clock = stream["clock"]
        measured = (frame["device_ticks"] - clock["first_tick"]) / clock["tick_hz"]
        expected = first / stream["sample_rate_hz"]
        residual = abs(measured - expected)
        self.maximum_clock_residual_s = max(self.maximum_clock_residual_s, residual)
        if residual > .05 + 1e-12:
            raise ValueError("sample-counter/device-clock residual exceeds50ms guard")
        if gap:
            self._log({"event": "sample_gap", "stream": stream["name"], "first_sample": state["next_sample"], "end_sample": first,
                       "start_s": stream["start_time_s"] + state["next_sample"] / stream["sample_rate_hz"],
                       "end_s": stream["start_time_s"] + first / stream["sample_rate_hz"], "missing_frames": lost_frames})
            zeros = np.zeros((gap, len(stream["channels"])), dtype=np.int32)
            missing = np.zeros(gap, dtype=bool)
            self.writer.append(stream["name"] + "_raw", zeros, missing)
            self.writer.append(stream["name"], np.full(zeros.shape, np.nan), missing)
        valid = frame["flags"] == 0
        physical = frame["samples"] * np.asarray(stream["scale"]) + np.asarray(stream["offset"])
        if not np.isfinite(physical).all():
            raise ValueError("scaling overflow")
        self.writer.append(stream["name"] + "_raw", frame["samples"], valid)
        self.writer.append(stream["name"], physical, valid)
        self.last_frames[frame["stream_id"]] = frame["frame_sha256"]
        state.update(next_sample=first + count, next_sequence=frame["sequence"] + 1, frames=state["frames"] + 1,
                     missing_samples=state["missing_samples"] + gap)
        self._log({"event": "frame_committed", **{key: frame[key] for key in ("stream_id", "sequence", "first_sample", "device_ticks", "boot_id", "device_session_id", "frame_sha256")},
                   "samples": count, "invalid_samples": int((~valid).sum()), "clock_residual_s": residual})

    def stop(self):
        if self.closed or self.failed:
            raise ValueError("recorder stopped or failed")
        if self.decoder.pending:
            self.abort("truncated final frame")
            raise ValueError("truncated final frame")
        try:
            self._log({"event": "capture_stopped", "transport_sha256": self.transport_hash.hexdigest(),
                       "maximum_clock_residual_s": self.maximum_clock_residual_s,
                       "independent_clock_alignment_verified": False, "streams": self.counters})
            self.writer.manifest["metadata"]["acquisition_config"]["transport_sha256"] = self.transport_hash.hexdigest()
            result = self.writer.finalize([])
        except BaseException as exc:
            self.abort(exc); raise
        self.transport.close(); self.log.close(); self.closed = True
        return result
