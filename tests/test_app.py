"""Phase 0 smoke tests: page served, health endpoint up, /ws/call echoes text."""
from fastapi.testclient import TestClient

from server.app import app


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


def test_ws_call_echoes_binary_frames() -> None:
    # Phase 1: the browser sends 20 ms PCM16 frames (640 bytes) as binary messages;
    # the echo path must return them byte-identical and in order.
    frame_a = bytes(range(256)) * 2 + bytes(128)  # 640 bytes, non-trivial payload
    frame_b = bytes(reversed(frame_a))
    client = TestClient(app)
    with client.websocket_connect("/ws/call") as ws:
        ws.send_bytes(frame_a)
        assert ws.receive_bytes() == frame_a
        ws.send_bytes(frame_b)
        assert ws.receive_bytes() == frame_b


def test_ws_call_echoes_text() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws/call") as ws:
        ws.send_text("hello")
        assert ws.receive_text() == "hello"
        ws.send_text("second frame")
        assert ws.receive_text() == "second frame"
