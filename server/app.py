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
the mount serve static/index.html for "/". /ws/call is a bare text echo in Phase 0:
it proves the WebSocket upgrade path through uvicorn works before any audio exists;
Phase 1 replaces the loop body with binary frame handling.
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
                text = await ws.receive_text()
                await ws.send_text(text)
        except WebSocketDisconnect:
            pass

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
