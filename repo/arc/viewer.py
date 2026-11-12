"""Loopback-only view of committed recording snapshots and offline research scores."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .pipeline import load_model, predict
from .session import Session


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def snapshot(path, end_s=None, window_s=10.):
    """Read at most 30 seconds per stream; short gaps survive plot downsampling."""
    window_s = _finite(window_s, "window_s")
    if not .1 <= window_s <= 30:
        raise ValueError("window_s must be between 0.1 and 30 seconds")
    session = Session(path, verify=False)
    end = session.duration_s if end_s is None else _finite(end_s, "end_s")
    if not 0 <= end <= session.duration_s:
        raise ValueError("end_s must be inside the saved recording")
    start = max(0., end - window_s)
    result = {"session_id": session.metadata["session_id"], "participant_id": session.metadata["participant_id"],
              "synthetic": session.metadata["synthetic"], "recording_status": session.recording_status,
              "complete": session.complete, "duration_s": session.duration_s, "start_s": start, "end_s": end,
              "saved_sample_count": sum(stream["n_samples"] for stream in session.streams.values()),
              "manifest_digest": session.content_digest, "preview_checksums_verified": False,
              "seconds_since_checkpoint": max(0., time.time() - (session.path / "session.json").stat().st_mtime),
              "alignment": session.metadata["acquisition_config"].get("clock_alignment",
                            session.metadata["acquisition_config"].get("reference_alignment")), "streams": []}
    if end == start:
        return result
    for name, descriptor in session.streams.items():
        if (end - start) * descriptor["sample_rate_hz"] * len(descriptor["channels"]) > 2_000_000:
            raise ValueError("Preview window exceeds two million channel samples; shorten window_s")
        samples, valid = session.read(name, start, end)
        rate, origin = descriptor["sample_rate_hz"], descriptor["start_time_s"]
        first = math.ceil((start - origin) * rate - 1e-9)
        times = origin + (first + np.arange(len(samples))) / rate
        edges = np.diff(np.r_[False, ~valid, False].astype(np.int8))
        gaps = [[max(start, origin + (first + left) / rate), min(end, origin + (first + right) / rate)]
                for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
        channels = []
        # Min/max envelopes keep signal excursions visible without sending full-rate audio.
        bounds = np.linspace(0, len(samples), min(len(samples), 600) + 1, dtype=int)
        for index, channel in enumerate(descriptor["channels"]):
            points = []
            for left, right in zip(bounds[:-1], bounds[1:]):
                selected = np.flatnonzero(valid[left:right]) + left
                if not len(selected):
                    points.append([float(times[left]), None])
                    continue
                values = samples[selected, index]
                chosen = sorted(set((int(selected[np.argmin(values)]), int(selected[np.argmax(values)]))))
                points.extend([[float(times[position]), float(samples[position, index])] for position in chosen])
            channels.append({"name": channel, "unit": descriptor["units"][index], "points": points})
        result["streams"].append({"name": name, "sample_rate_hz": rate, "sample_count": descriptor["n_samples"],
                                  "window_samples": len(valid), "invalid_window_samples": int((~valid).sum()),
                                  "missing_ranges": gaps, "channels": channels})
    return result


class RecordingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, session_path, model_path):
        self.session_path = Path(session_path).resolve()
        self.model_path = Path(model_path).resolve() if model_path else None
        self.score_lock = threading.Lock()
        self.score_state = {"status": "idle", "model_configured": bool(model_path)}
        self.score_file = None
        super().__init__(address, RecordingHandler)

    def score(self):
        temporary = None
        try:
            session = Session(self.session_path)
            if not session.complete:
                raise ValueError("Stop and finalize the recording before offline scoring; recovered sessions stay incomplete.")
            model = load_model(self.model_path)
            fd, temporary = tempfile.mkstemp(prefix=f"{session.path.name}-score-", suffix=".json.tmp", dir=session.path.parent)
            count = invalid = 0
            preview = []
            with os.fdopen(fd, "w") as output:
                metadata = {"intended_use": "research_only", "synthetic_only": session.metadata["synthetic"],
                            "model_version": model["model_version"], "model_digest": model["bundle_digest"],
                            "task": model["task"], "session_id": session.metadata["session_id"],
                            "session_digest": session.content_digest}
                output.write(json.dumps(metadata, allow_nan=False)[:-1] + ',"predictions":[')
                for row in predict(session, model):
                    if count:
                        output.write(",")
                    json.dump(row, output, allow_nan=False)
                    if len(preview) < 12:
                        preview.append(row)
                    count += 1
                    invalid += int(not row["valid"])
                output.write("]}\n")
                output.flush()
                os.fsync(output.fileno())
            final = Path(temporary).with_suffix("")
            os.rename(temporary, final)
            temporary = None
            with self.score_lock:
                self.score_file = final
                self.score_state = {"status": "complete", "model_configured": True, "prediction_bins": count,
                                    "invalid_bins": invalid, "preview": preview, "model_version": model["model_version"],
                                    "synthetic": session.metadata["synthetic"], "download": "/api/score/download"}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            with self.score_lock:
                self.score_state = {"status": "error", "model_configured": bool(self.model_path), "error": str(exc)}
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)


class RecordingHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _allowed(self, mutation=False):
        if len(self.headers.get_all("Host", [])) != 1:
            self._json({"error": "Exactly one Host header required"}, 403)
            return False
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in hosts or self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json({"error": "Loopback access only"}, 403)
            return False
        if (mutation and origin != f"http://{host}") or (origin and origin != f"http://{host}"):
            self._json({"error": "Same-origin request required"}, 403)
            return False
        return True

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

    def _json(self, result, status=200):
        data = json.dumps(result, allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def do_GET(self):
        if not self._allowed():
            return
        route = urlsplit(self.path)
        try:
            if route.path == "/" and not route.query:
                data = Path(__file__).with_name("viewer.html").read_bytes()
                self._headers(200, "text/html; charset=utf-8", len(data))
                self.wfile.write(data)
            elif route.path == "/api/session":
                arguments = parse_qs(route.query, keep_blank_values=True)
                if set(arguments) - {"end_s", "window_s"} or any(len(values) != 1 for values in arguments.values()):
                    raise ValueError("Unknown or repeated time-window parameter")
                result = snapshot(self.server.session_path, arguments.get("end_s", [None])[0], arguments.get("window_s", [10.])[0])
                result["model_configured"] = bool(self.server.model_path)
                self._json(result)
            elif route.path == "/api/score" and not route.query:
                with self.server.score_lock:
                    self._json(self.server.score_state)
            elif route.path == "/api/score/download" and not route.query:
                with self.server.score_lock:
                    path = self.server.score_file
                if path is None:
                    self._json({"error": "No completed offline score is available"}, 404)
                    return
                with path.open("rb") as output:
                    self._headers(200, "application/json", path.stat().st_size)
                    for block in iter(lambda: output.read(65536), b""):
                        self.wfile.write(block)
            else:
                self._json({"error": "Not found"}, 404)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._json({"error": str(exc)}, 400)

    def do_POST(self):
        if not self._allowed(mutation=True):
            return
        if self.path != "/api/score":
            self._json({"error": "Not found"}, 404)
            return
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0") != "0":
            self._json({"error": "This action accepts no request body"}, 400)
            return
        if self.server.model_path is None:
            self._json({"error": "Start the viewer with --model to enable offline scoring"}, 409)
            return
        try:
            if not Session(self.server.session_path, verify=False).complete:
                self._json({"error": "Only finalized complete sessions can be scored"}, 409)
                return
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        with self.server.score_lock:
            if self.server.score_state["status"] == "running":
                self._json({"error": "Offline scoring is already running"}, 409)
                return
            self.server.score_file = None
            self.server.score_state = {"status": "running", "model_configured": True}
        threading.Thread(target=self.server.score, daemon=True).start()
        self._json({"status": "running"}, 202)


def create_server(path, host="127.0.0.1", port=8765, model_path=None):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("The recording viewer binds only to loopback")
    Session(path, verify=False)
    if model_path:
        load_model(model_path)
    return RecordingServer(("127.0.0.1", port), path, model_path)


def serve_session(path, host="127.0.0.1", port=8765, model_path=None):
    server = create_server(path, host, port, model_path)
    print(f"Arc recording view: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
