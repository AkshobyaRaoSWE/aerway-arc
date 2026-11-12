"""Offline replay evidence from recorded samples, without interpolation or pacing."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def iter_replay(session, stream, block_s=1.0):
    """Yield bounded original samples, validity and absolute session timestamps.

    Blocks may end at stored chunk boundaries. No real-time sleeping, resampling,
    synthetic gap filling or invalid-value replacement occurs in this export.
    """
    _positive(block_s, "block_s")
    descriptor = session.streams[stream]
    rate, origin = descriptor["sample_rate_hz"], descriptor["start_time_s"]
    size = descriptor["n_samples"] if block_s >= descriptor["n_samples"] / rate else max(1, int(block_s * rate))
    for first, samples, valid in session.iter_chunks(stream):
        for offset in range(0, len(samples), size):
            last = min(offset + size, len(samples))
            indices = np.arange(first + offset, first + last, dtype=np.int64)
            yield {"sample_index": indices, "time_s": origin + indices / rate,
                   "samples": samples[offset:last], "valid": valid[offset:last]}


def _append_invalid(ranges, first, mask):
    padded = np.r_[False, ~mask, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1) + first
    ends = np.flatnonzero(np.diff(padded) == -1) + first
    for start, end in zip(starts.tolist(), ends.tolist()):
        if ranges and ranges[-1][1] == start:
            ranges[-1][1] = end
        else:
            ranges.append([start, end])


def _ranges(descriptor, invalid, duration_s):
    rate, origin = descriptor["sample_rate_hz"], descriptor["start_time_s"]
    ranges = [{"first_sample": first, "end_sample": end, "start_s": origin + first / rate,
               "end_s": origin + end / rate, "kind": "invalid_samples"} for first, end in invalid]
    if origin > 0:
        ranges.insert(0, {"first_sample": None, "end_sample": None, "start_s": 0.,
                          "end_s": origin, "kind": "not_captured"})
    end = origin + descriptor["n_samples"] / rate
    if end < duration_s:
        ranges.append({"first_sample": None, "end_sample": None, "start_s": end,
                       "end_s": duration_s, "kind": "not_captured"})
    return ranges


def _digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def export_replay(session, output_dir, block_s=1.0):
    """Write exact numeric CSV replay plus channel/unit/gap evidence.

    Invalid sample values remain verbatim in CSV with valid=0. Consumers must
    apply that mask. replay-audit.json is written only after every CSV completes.
    Source files are never changed; existing export directories are refused.
    """
    _positive(block_s, "block_s")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    audit = {"schema_version": 1, "session_id": session.metadata["session_id"],
             "source_manifest_sha256": session.content_digest,
             "synthetic_only": session.metadata["synthetic"],
             "recording_status": getattr(session, "recording_status", "complete"),
             "duration_s": session.duration_s, "replay_mode": "offline original sample order, not wall-clock playback",
             "invalid_values": "preserved in CSV; apply valid=0 mask before analysis",
             "timestamp_definition": "start_time_s + sample_index / sample_rate_hz",
             "streams": {}}
    for name, descriptor in session.streams.items():
        filename = f"{name}.csv"
        invalid, count, valid_count = [], 0, 0
        with (output / filename).open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "time_s", *[f"value:{channel}" for channel in descriptor["channels"]], "valid"])
            for block in iter_replay(session, name, block_s):
                for index, timestamp, values, valid in zip(block["sample_index"], block["time_s"], block["samples"], block["valid"]):
                    writer.writerow([int(index), float(timestamp), *values.tolist(), int(valid)])
                _append_invalid(invalid, int(block["sample_index"][0]), block["valid"])
                count += len(block["valid"])
                valid_count += int(block["valid"].sum())
        if count != descriptor["n_samples"]:
            raise ValueError("replay sample count differs from source manifest")
        audit["streams"][name] = {
            "file": filename, "sha256": _digest(output / filename), "sample_count": count,
            "valid_sample_count": valid_count, "invalid_sample_count": count - valid_count,
            "sample_rate_hz": descriptor["sample_rate_hz"], "start_time_s": descriptor["start_time_s"],
            "end_time_s": descriptor["start_time_s"] + count / descriptor["sample_rate_hz"],
            "channels": descriptor["channels"], "units": descriptor["units"],
            "value_columns": [f"value:{channel}" for channel in descriptor["channels"]],
            "missing_ranges": _ranges(descriptor, invalid, session.duration_s),
        }
    with (output / "replay-audit.json").open("x") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return audit


def _preview(session, name, duration, max_points):
    descriptor = session.streams[name]
    rate, origin = descriptor["sample_rate_hz"], descriptor["start_time_s"]
    if duration >= origin + descriptor["n_samples"] / rate:
        limit = descriptor["n_samples"]
    elif duration <= origin:
        limit = 0
    else:
        limit = math.ceil((duration - origin) * rate - 1e-9)
    stride = max(1, math.ceil(limit / max_points))
    times, values, invalid = [], [], []
    for block in iter_replay(session, name):
        selected = block["sample_index"] < limit
        if not selected.any():
            break
        indices, mask = block["sample_index"][selected], block["valid"][selected]
        _append_invalid(invalid, int(indices[0]), mask)
        keep = indices % stride == 0
        times.extend(block["time_s"][selected][keep].tolist())
        data = block["samples"][selected][keep].astype(float)
        data[~mask[keep]] = np.nan
        values.extend(data.tolist())
    ranges = _ranges(descriptor, invalid, duration)
    times = np.asarray(times)
    values = np.asarray(values).reshape(-1, len(descriptor["channels"]))
    # Keep even a one-sample dropout visible when waveform decimation skips it.
    for span in reversed(ranges):
        position = int(np.searchsorted(times, span["start_s"]))
        times = np.insert(times, position, span["start_s"])
        values = np.insert(values, position, np.nan, axis=0)
    return times, values, ranges


def plot_session(session, output_path, max_points=12000, max_duration_s=300.):
    """Plot a bounded preview; full-mask gap spans survive waveform decimation."""
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 2:
        raise ValueError("max_points must be an integer >= 2")
    _positive(max_duration_s, "max_duration_s")
    if not session.streams or session.duration_s <= 0:
        raise ValueError("no committed samples available to plot")
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("plot output must be a .png file")
    if path.exists():
        raise FileExistsError(path)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    count = sum(len(item["channels"]) for item in session.streams.values())
    duration = min(session.duration_s, max_duration_s)
    settings = {"figure.facecolor": "#121212", "axes.facecolor": "#121212", "text.color": "#eeeeee",
                "axes.labelcolor": "#cccccc", "xtick.color": "#aaaaaa", "ytick.color": "#aaaaaa",
                "axes.edgecolor": "#555555", "font.size": 9, "savefig.facecolor": "#121212"}
    with plt.rc_context(settings):
        figure, axes = plt.subplots(count, 1, figsize=(12, 1.9 * count + 1.1), sharex=True, squeeze=False)
        index = 0
        for name, descriptor in session.streams.items():
            times, values, ranges = _preview(session, name, duration, max_points)
            for channel, unit in zip(range(len(descriptor["channels"])), descriptor["units"]):
                axis = axes[index, 0]
                axis.plot(times, values[:, channel], color="#dddddd", linewidth=.6)
                for span in ranges:
                    if span["start_s"] < duration:
                        axis.axvspan(max(0., span["start_s"]), min(duration, span["end_s"]),
                                     facecolor="#666666", edgecolor="#999999", alpha=.4, hatch="///", linewidth=.4)
                axis.set_ylabel(unit)
                axis.set_title(f"{name} / {descriptor['channels'][channel]} / {descriptor['sample_rate_hz']:g} Hz",
                               loc="left", fontsize=10)
                axis.grid(axis="y", color="#333333", linewidth=.4)
                axis.spines[["top", "right"]].set_visible(False)
                index += 1
        axes[-1, 0].set_xlabel("Session time (s)")
        axes[-1, 0].set_xlim(0, duration)
        evidence = "SYNTHETIC SOFTWARE CHECK" if session.metadata["synthetic"] else "RECORDED SESSION"
        status = getattr(session, "recording_status", "complete")
        figure.suptitle(f"{evidence} | {session.metadata['session_id']} | {status}\n"
                        f"Preview 0-{duration:g} s of {session.duration_s:g} s; shaded = invalid/not captured; waveforms may be decimated",
                        fontsize=11, y=.99)
        figure.tight_layout(rect=(0, 0, 1, .94))
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                figure.savefig(handle, format="png", dpi=160)
        finally:
            plt.close(figure)
    return {"path": str(path), "preview_duration_s": duration, "session_duration_s": session.duration_s,
            "synthetic_only": session.metadata["synthetic"], "recording_status": status}
