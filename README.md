# Screener — a from-scratch voice screening agent

A local browser Voice AI prototype built with **no voice platform and no agent framework**:
browser mic → WebSocket → FastAPI → VAD → streaming ASR (Deepgram)
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

## Implemented architecture
```
browser mic ──ws──▶ /ws/call ─▶ recv → VAD ─▶ ASR task (Deepgram ws) ─▶ endpointer
                                                                          │ turn_complete
                                                                          ▼
        audio ◀── TTS task (ElevenLabs) ◀── sentence chunker ◀── engine (OpenAI stream + tools)
        generation supervisor invalidates, cancels, and awaits before replacement
        bounded playback acks clear/drained/overflow with the same generation id
        on hangup → caller-utterance evidence verification → deterministic score/report
```

The ONNX VAD session is process-shared, while recurrent state, context, frame
carry, gate state, and reset lifetime are owned by each call. Reply generations
are per-call and monotonic: stale audio, tool effects, transcript updates, and
delayed playback acknowledgements are rejected by ownership checks.

Each WebSocket receives one stable call ID and follows an explicit consent-first
session lifecycle. The agent initiates with truthful transcription/analysis
disclosure; no audio recording is stored. Consent refusal ends without extraction.
Reports and metrics stay keyed to that call ID. Numeric scores and knockouts use
only verified caller utterances; uncertain material evidence requires human review.

## Metrics that matter (check /metrics/{call_id} after a call)
endpoint_delay, llm_ttft, tts_ttfb, turn_latency (p50/p95), barge-ins, turns.
