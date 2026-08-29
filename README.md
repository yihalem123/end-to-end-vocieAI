# Screener — a from-scratch voice screening agent

A production-shaped voice AI agent built with **no voice platform and no agent framework**:
browser mic (later: Twilio phone call) → WebSocket → FastAPI → VAD → streaming ASR (Deepgram)
→ interview-plan engine (OpenAI, streaming + tools) → streaming TTS (ElevenLabs) → back to the caller.
Includes barge-in, per-turn latency metrics, post-call evidence extraction, and deterministic scoring.

Built to demonstrate ownership of the full real-time loop. Development is co-piloted with Claude Code:
see `PLAN.md` for the phase-by-phase build and `CLAUDE.md` for project conventions.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in your keys
uvicorn server.app:app --reload --port 8080
# open http://localhost:8080  → the test console
```

## Architecture (target)
```
browser mic ──ws──▶ /ws/call ─▶ recv → VAD ─▶ ASR task (Deepgram ws) ─▶ endpointer
phone (Twilio) ──ws──▶ /ws/twilio ─┘                                      │ turn_complete
                                                                          ▼
        audio ◀── TTS task (ElevenLabs) ◀── sentence chunker ◀── engine (OpenAI stream + tools)
        barge-in controller cancels engine+TTS on speech onset · CallState persisted per turn
        on hangup → postcall: extraction (quotes+confidence) → deterministic score → report
```

## Metrics that matter (check /metrics after a call)
endpoint_delay, llm_ttft, tts_ttfb, turn_latency (p50/p95), barge-ins, turns.
