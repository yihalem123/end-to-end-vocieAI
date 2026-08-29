"""FastAPI app factory. Phases 0-2.

Endpoints:
  GET  /            -> static/index.html test console
  GET  /healthz
  WS   /ws/call     -> the call pipeline: VAD + ASR + endpointer (Phase 2)
  WS   /ws/echo     -> byte-identical echo, kept as a latency diagnostic (Phase 1)
  GET  /metrics     -> per-stage latency percentiles (Phase 3)

## How this works
create_app(settings) builds the app; tests inject Settings(_env_file=None, ...) to
avoid touching real keys, while the module-level `app` (what uvicorn imports)
reads .env. Route registration order matters: specific routes are declared before
the StaticFiles mount at "/" because the mount is a catch-all. /ws/call hands the
socket straight to CallSession (server/realtime/call.py) — the app layer stays
transport-only; pipeline logic lives with the pipeline. The echo handler uses raw
ws.receive() because only it accepts both text and binary in one loop; raw
receive() reports disconnect as a message type, not an exception.
"""
import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from server.config import Settings, get_settings
from server.metrics import registry
from server.postcall.report import render_html, reports as report_store, run_postcall
from server.realtime.call import CallSession

_postcall_tasks: set[asyncio.Task] = set()  # keep refs; tasks self-remove

log = logging.getLogger(__name__)

# uvicorn configures its own loggers but leaves root at WARNING; without this,
# the pipeline's log.info lines (call summaries, reconnects) are invisible.
logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="screener")
    app.state.settings = settings if settings is not None else get_settings()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict:
        return registry.snapshot()

    @app.get("/calls")
    async def calls() -> list[dict]:
        return [{"call_id": cid,
                 "score": rep.get("score"),
                 "needs_review": rep.get("needs_review"),
                 "knocked_out": rep.get("knocked_out"),
                 "created_at": rep.get("created_at")}
                for cid, rep in reversed(report_store.items())]

    @app.get("/report/{call_id}")
    async def report_json(call_id: str) -> dict:
        if call_id not in report_store:
            raise HTTPException(status_code=404, detail="no such call")
        return report_store[call_id]

    @app.get("/report/{call_id}/view")
    async def report_view(call_id: str) -> HTMLResponse:
        if call_id not in report_store:
            raise HTTPException(status_code=404, detail="no such call")
        return HTMLResponse(render_html(report_store[call_id]))

    @app.websocket("/ws/call")
    async def ws_call(ws: WebSocket) -> None:
        session = CallSession(ws, app.state.settings)
        try:
            await session.run()
        except Exception:
            # A vendor failure must not take uvicorn down with a bare traceback;
            # log it and close the socket so the client sees a clean end.
            log.exception("call session failed")
            try:
                await ws.close(code=1011)
            except RuntimeError:
                pass  # already closed
        finally:
            state = session.state
            if state.conversation and app.state.settings.openai_api_key:
                call_id = uuid.uuid4().hex[:12]
                task = asyncio.create_task(run_postcall(
                    call_id, state.conversation, state.turns, app.state.settings))
                _postcall_tasks.add(task)
                task.add_done_callback(_postcall_tasks.discard)
                log.info("postcall started: /report/%s/view", call_id)

    @app.websocket("/ws/echo")
    async def ws_echo(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                message = await ws.receive()
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
