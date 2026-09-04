"""FastAPI app factory.

Endpoints:
  GET  /             -> static/index.html test console
  GET  /healthz
  POST /prewarm      -> open a TTS socket before a call exists
  WS   /ws/call      -> the browser call pipeline: VAD + ASR + endpointer + replies
  WS   /ws/twilio    -> the same pipeline behind the Twilio Media Streams adapter
  POST /twilio/voice -> TwiML that connects an incoming call to /ws/twilio
  WS   /ws/echo      -> byte-identical echo, kept as a latency diagnostic
  GET  /metrics      -> per-stage latency percentiles; /report/{id}[/view]

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
from contextlib import asynccontextmanager
import uuid
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from server.config import Settings, get_settings
from server.metrics import registry
from server.postcall.report import (
    render_html,
    reports as report_store,
    run_postcall,
    store_terminal_report,
)
from server.engine.plan import load_plan_cached
from server.realtime.call import CallSession
from server.realtime.session import SessionStatus
from server.realtime import tts_aura
from server.realtime.twilio import TwilioSocket, twiml_connect_stream, verify_twilio_signature
from server.realtime.vad import SileroRuntime

_postcall_tasks: set[asyncio.Task] = set()  # keep refs; tasks self-remove

log = logging.getLogger(__name__)

# uvicorn configures its own loggers but leaves root at WARNING; without this,
# the pipeline's log.info lines (call summaries, reconnects) are invisible.
logging.basicConfig(level=logging.INFO)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Fail fast at boot: a broken plan file or missing VAD model must
        # surface at startup, not on call #1. Both loads are cached, so the
        # first call pays nothing.
        # Both operations perform blocking file/CPU work. Startup is the right
        # lifecycle boundary, but it is still an async context: keep its event
        # loop responsive for sibling startup tasks.
        app.state.plan, app.state.vad_runtime = await asyncio.gather(
            asyncio.to_thread(load_plan_cached, str(resolved.plan_path)),
            asyncio.to_thread(SileroRuntime),
        )
        log.info("startup warm: plan %r validated, vad runtime loaded",
                 resolved.plan_path)
        yield
        await tts_aura.warm_sockets.close()

    app = FastAPI(title="screener", lifespan=lifespan)
    app.state.settings = resolved

    @app.post("/prewarm", status_code=204)
    async def prewarm() -> Response:
        """Open a TTS socket before a call exists.

        The console calls this on page load: the Aura connect measured a median
        2377 ms and is otherwise paid on the greeting, the first thing anyone
        hears. Fire-and-forget — a failed warm just means the call connects on
        demand, exactly as it did before."""
        if resolved.tts_provider == "aura" and resolved.deepgram_api_key:
            tts_aura.warm_sockets.prewarm(
                tts_aura.build_aura_url(resolved.aura_model),
                resolved.deepgram_api_key)
        return Response(status_code=204)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict:
        return registry.snapshot()

    @app.get("/metrics/{call_id}")
    async def call_metrics(call_id: str) -> dict:
        return registry.snapshot(call_id)

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

    async def _run_call(sock, mode: str, call_id: str) -> None:
        """One call runner for every transport: the browser socket and the
        Twilio adapter both present the same receive()/send contract."""
        session = CallSession(
            sock, app.state.settings, call_id=call_id, mode=mode,
            plan=getattr(app.state, "plan", None),
            vad_runtime=getattr(app.state, "vad_runtime", None),
        )
        try:
            await session.run()
        except Exception:
            # A vendor failure must not take uvicorn down with a bare traceback;
            # log it and close the socket so the client sees a clean end.
            log.exception("call session failed")
            if session.state.session.status not in {
                SessionStatus.CONSENT_REFUSED, SessionStatus.COMPLETED,
                SessionStatus.FAILED, SessionStatus.CANCELLED,
            }:
                session.state.session.transition(SessionStatus.FAILED)
            try:
                await sock.close(code=1011)
            except RuntimeError:
                pass  # already closed
        finally:
            state = session.state
            status = state.session.status
            if status in {SessionStatus.DISCLOSURE, SessionStatus.AWAITING_CONSENT}:
                state.session.transition(SessionStatus.CANCELLED)
                store_terminal_report(call_id, state.conversation,
                                      state.session.status, "call ended before consent")
            elif status == SessionStatus.CONSENT_REFUSED:
                store_terminal_report(call_id, state.conversation, status,
                                      "candidate declined consent; no analysis performed")
            elif status == SessionStatus.FAILED:
                store_terminal_report(call_id, state.conversation, status,
                                      "call failed before completion")
            elif status in {SessionStatus.INTERVIEWING, SessionStatus.CLOSING}:
                if not app.state.settings.openai_api_key:
                    state.session.transition(SessionStatus.FAILED)
                    store_terminal_report(call_id, state.conversation,
                                          state.session.status,
                                          "post-call extraction unavailable")
                else:
                    state.session.transition(SessionStatus.POST_PROCESSING)
                    task = asyncio.create_task(run_postcall(
                        call_id, state.conversation, state.turns, app.state.settings,
                        lifecycle=state.session,
                        tool_ledger=session.tool_ledger))
                    _postcall_tasks.add(task)
                    task.add_done_callback(_postcall_tasks.discard)
                    log.info("postcall started: /report/%s/view", call_id)

    @app.websocket("/ws/call")
    async def ws_call(ws: WebSocket) -> None:
        await _run_call(ws, ws.query_params.get("mode", "custom"), uuid.uuid4().hex)

    @app.websocket("/ws/twilio")
    async def ws_twilio(ws: WebSocket) -> None:
        # The phone leg: Twilio Media Streams wrapped to look like the browser.
        call_id = uuid.uuid4().hex
        await _run_call(TwilioSocket(ws, call_id), "custom", call_id)

    @app.api_route("/twilio/voice", methods=["GET", "POST"])
    async def twilio_voice(request: Request) -> Response:
        """TwiML for an incoming (or REST-placed) call: connect its audio here."""
        form = (dict(await request.form()) if request.method == "POST"
                else dict(request.query_params))
        if resolved.twilio_auth_token:
            url = (resolved.public_base_url.rstrip("/") + request.url.path
                   if resolved.public_base_url else str(request.url))
            if not verify_twilio_signature(url, {k: str(v) for k, v in form.items()},
                                           resolved.twilio_auth_token,
                                           request.headers.get("X-Twilio-Signature")):
                raise HTTPException(status_code=403, detail="bad twilio signature")
        origin = resolved.public_base_url or f"https://{request.headers.get('host', 'localhost')}"
        ws_url = origin.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
        return Response(twiml_connect_stream(ws_url + "/ws/twilio"), media_type="text/xml")

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

    class NoStoreStatic(StaticFiles):
        """The console's JS *is* the app. A cached audio.js or worklet runs
        yesterday's code against today's server and reads as "the fix didn't
        work" — it cost one live debugging session already. This is a local
        dev console, so correctness beats the handful of saved requests."""

        def file_response(self, *args: Any, **kwargs: Any) -> Response:
            response = super().file_response(*args, **kwargs)
            response.headers["cache-control"] = "no-store, must-revalidate"
            return response

    app.mount("/", NoStoreStatic(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
