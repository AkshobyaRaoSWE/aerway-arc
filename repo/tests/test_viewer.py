import json
from http.client import HTTPConnection
import threading
import time

import numpy as np
import pytest

from arc import viewer
from arc.session import SessionWriter, write_session


def recording(path, complete=True):
    metadata = dict(participant_id="bench-fixture", session_id="viewer-fixture", placement_id="bench",
                    source="synthetic test", source_version="1", permission="generated fixture",
                    synthetic=True, acquisition_config={"clock": "synthetic"}, label_source="synthetic")
    config = dict(sample_rate_hz=1000, start_time_s=.25, channels=["signal"], units=["count"])
    values = np.arange(2000, dtype=float).reshape(-1, 1)
    valid = np.ones(2000, dtype=bool)
    valid[1201] = False
    if complete:
        write_session(path, metadata, {"raw": dict(config, samples=values, valid=valid)}, [], chunk_samples=800)
    else:
        writer = SessionWriter(path, metadata, {"raw": config})
        return writer, values, valid
    return path


@pytest.fixture
def server(tmp_path):
    path = recording(tmp_path / "session")
    service = viewer.create_server(path, port=0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    yield service
    service.shutdown()
    service.server_close()
    thread.join()


def request(server, path, method="GET", headers=None, body=None):
    connection = HTTPConnection(*server.server_address, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_snapshot_preserves_subpixel_gap_and_native_timestamps(tmp_path):
    result = viewer.snapshot(recording(tmp_path / "session"))
    stream = result["streams"][0]
    assert [1.451, 1.452] in stream["missing_ranges"]
    assert stream["invalid_window_samples"] == 251
    assert stream["sample_count"] == 2000
    assert len(stream["channels"][0]["points"]) <= 1200
    assert stream["channels"][0]["unit"] == "count"
    assert result["preview_checksums_verified"] is False
    json.dumps(result, allow_nan=False)


def test_live_snapshot_advances_only_with_saved_chunks(tmp_path):
    path = tmp_path / "session"
    writer, values, valid = recording(path, complete=False)
    assert viewer.snapshot(path)["streams"] == []
    writer.append("raw", values[:100], valid[:100])
    first = viewer.snapshot(path)
    writer.append("raw", values[100:200], valid[100:200])
    second = viewer.snapshot(path)
    assert first["duration_s"] < second["duration_s"]
    assert second["recording_status"] == "recording" and not second["complete"]


@pytest.mark.parametrize("query", ["end_s=nan", "end_s=-1", "end_s=500", "window_s=inf", "window_s=31", "x=1", "end_s=1&end_s=2"])
def test_invalid_windows_rejected(server, query):
    assert request(server, "/api/session?" + query)[0] == 400


def test_http_is_readonly_fixed_path_same_origin(server):
    status, headers, content = request(server, "/")
    assert status == 200 and b"Recording view" in content
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert "Access-Control-Allow-Origin" not in headers
    assert request(server, "/../../etc/passwd")[0] == 404
    assert request(server, "/api/session", headers={"Host": "attacker.example"})[0] == 403
    assert request(server, "/api/session", headers={"Origin": "https://attacker.example"})[0] == 403
    assert request(server, "/api/session", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert request(server, "/api/score", "POST")[0] == 403
    origin = {"Origin": f"http://127.0.0.1:{server.server_address[1]}"}
    assert request(server, "/api/score", "POST", origin)[0] == 409
    assert request(server, "/api/score", "POST", origin, "{}")[0] == 400
    assert request(server, "/api/score/download")[0] == 404
    assert json.loads(request(server, "/api/session")[2])["complete"] is True


def test_scoring_uses_existing_pipeline_saves_sibling_and_surfaces_mismatch(server, monkeypatch):
    server.model_path = "configured-test-model"
    monkeypatch.setattr(viewer, "load_model", lambda path: {"model_version": "fixture-v1", "bundle_digest": "fixture-digest", "task": "night"})
    monkeypatch.setattr(viewer, "predict", lambda session, model: iter([{"valid": True, "probability": .3}]))
    origin = {"Origin": f"http://127.0.0.1:{server.server_address[1]}"}
    assert request(server, "/api/score", "POST", origin)[0] == 202
    for _ in range(100):
        state = json.loads(request(server, "/api/score")[2])
        if state["status"] != "running":
            break
        time.sleep(.01)
    assert state["status"] == "complete"
    report = json.loads(request(server, "/api/score/download")[2])
    assert report["predictions"][0]["probability"] == .3
    assert report["synthetic_only"] is True
    assert report["model_version"] == "fixture-v1" and report["model_digest"] == "fixture-digest"
    assert report["task"] == "night" and report["session_id"] == "viewer-fixture"
    assert report["session_digest"] == viewer.Session(server.session_path).content_digest
    assert report["intended_use"] == "research_only"
    assert server.score_file.parent == server.session_path.parent
    def mismatch(session, model):
        raise ValueError("synthetic model cannot score real data")
    monkeypatch.setattr(viewer, "predict", mismatch)
    server.score()
    assert server.score_state["status"] == "error"
    assert "synthetic model" in server.score_state["error"]


def test_non_loopback_bind_refused(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        viewer.create_server(tmp_path, host="0.0.0.0")
