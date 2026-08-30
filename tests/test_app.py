"""App-level smoke tests: page served, health endpoint, echo diagnostic, call guard."""
import json

from fastapi.testclient import TestClient

from server.app import app, create_app
from server.config import Settings


def test_healthz_ok() -> None:
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_page_served() -> None:
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_ws_echo_echoes_binary_frames() -> None:
    # /ws/echo is the latency diagnostic left over from Phase 1: byte-identical,
    # ordered echo of 640-byte frames.
    frame_a = bytes(range(256)) * 2 + bytes(128)
    frame_b = bytes(reversed(frame_a))
    client = TestClient(app)
    with client.websocket_connect("/ws/echo") as ws:
        ws.send_bytes(frame_a)
        assert ws.receive_bytes() == frame_a
        ws.send_bytes(frame_b)
        assert ws.receive_bytes() == frame_b


def test_ws_echo_echoes_text() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws/echo") as ws:
        ws.send_text("hello")
        assert ws.receive_text() == "hello"


def test_browser_forwards_playback_started_ack() -> None:
    client = TestClient(app)
    assert '"playback_started"' in client.get("/audio.js").text
    assert 'type: "playback_started"' in client.get("/playback-processor.js").text


def test_ws_call_without_key_reports_error() -> None:
    # The call pipeline needs Deepgram; with no key it must fail loudly and
    # immediately, not half-start.
    bare = create_app(Settings(_env_file=None, deepgram_api_key=""))
    client = TestClient(bare)
    with client.websocket_connect("/ws/call") as ws:
        session = json.loads(ws.receive_text())
        assert session["type"] == "session"
        assert len(session["call_id"]) == 32
        failed = json.loads(ws.receive_text())
        assert failed == {"type": "session_state", "state": "failed",
                          "call_id": session["call_id"]}
        error = json.loads(ws.receive_text())
        assert error["type"] == "error"
        assert error["call_id"] == session["call_id"]
        assert "DEEPGRAM_API_KEY" in error["message"]


def test_lifespan_warms_plan_and_vad_and_fails_fast_on_bad_plan(tmp_path) -> None:
    import pytest
    from server.app import create_app
    from server.config import Settings

    with TestClient(create_app(Settings(_env_file=None))) as client:
        assert client.app.state.plan.scoring_version  # warmed at startup

    bad = Settings(_env_file=None, plan_path=str(tmp_path / "missing.yaml"))
    with pytest.raises(Exception):
        with TestClient(create_app(bad)):
            pass  # startup itself must fail, not call #1


def test_console_assets_are_never_cached() -> None:
    # The console's JS IS the app: a cached audio.js or capture-processor.js
    # silently runs yesterday's code against today's server. That cost a live
    # debugging session (a fixed mic worklet appeared still broken), and it is
    # exactly how a demo take goes wrong.
    with TestClient(create_app(Settings(_env_file=None))) as client:
        for path in ("/", "/audio.js", "/capture-processor.js"):
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert "no-store" in resp.headers.get("cache-control", ""), path
