"""FastAPI app factory. Built in Phase 0 (see PLAN.md).

Endpoints (target):
  GET  /            -> static/index.html test console
  GET  /healthz
  WS   /ws/call     -> browser leg (16k PCM16 frames)   [Phase 1+]
  GET  /metrics     -> per-stage latency percentiles     [Phase 3]

## How this works
create_app() builds the app so tests can construct fresh instances; the module-level
`app` is what `uvicorn server.app:app` imports. Route registration order matters:
/healthz and /ws/call are declared before the StaticFiles mount at "/" because the
mount is a catch-all — anything declared after it would be shadowed. html=True makes
the mount serve static/index.html for "/". /ws/call (Phase 1) echoes both message
kinds: binary frames are the audio path (640-byte 20 ms PCM16 from the browser);
text remains for control/debug. We use the low-level ws.receive() instead of
receive_text()/receive_bytes() because only it lets one loop accept either kind —
the typed helpers raise on a mismatched message type.
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="screener")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/call")
    async def ws_call(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                message = await ws.receive()
                # Raw receive() does NOT raise WebSocketDisconnect (only the typed
                # helpers do) — the disconnect arrives as a message we must handle.
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await ws.send_bytes(message["bytes"])
                elif message.get("text") is not None:
                    await ws.send_text(message["text"])
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
