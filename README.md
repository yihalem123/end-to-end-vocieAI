# Screener — a from-scratch voice screening agent

A local browser Voice AI prototype built with **no voice platform and no agent framework**:
browser mic → WebSocket → FastAPI → VAD → streaming ASR (Deepgram)
→ bounded objective engine (OpenAI, speech + concurrent extraction) → streaming TTS (Deepgram
Aura or ElevenLabs, selectable via `TTS_PROVIDER`) → back to the caller.
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
        audio ◀── TTS task ◀── sentence chunker ◀── tool-free OpenAI speech stream
                                                     └── committed-turn extraction tools
        generation supervisor invalidates, cancels, and awaits before replacement
        speculative speech/audio are promotion-gated; extraction starts only on commit
        committed caller evidence applies in turn order and never gates speech
        reliable events outrank one replaceable latest-partial slot
        bounded playback acks clear/drained/overflow with the same generation id
        on hangup → caller-utterance evidence verification → deterministic score/report
```

The ONNX VAD session is process-shared, while recurrent state, context, frame
carry, gate state, and reset lifetime are owned by each call. Reply generations
are per-call and monotonic: stale audio, transcript updates, and delayed playback
acknowledgements are rejected by ownership checks. Evidence extraction belongs
to immutable committed caller turns, so interrupting agent playback cannot erase
a real answer; speculative text can never mutate evidence.

Each WebSocket receives one stable call ID and follows an explicit consent-first
session lifecycle. The agent initiates with truthful transcription/analysis
disclosure; no audio recording is stored. Consent refusal ends without extraction.
Reports and metrics stay keyed to that call ID. Numeric scores and knockouts use
only verified caller utterances; uncertain material evidence requires human review.

Two turn-taking stacks are selectable per call from the console ("Flux
end-of-turn" toggle): the custom VAD-anchored endpointer (three patience tiers)
or Deepgram Flux's model-integrated end-of-turn. Both can prepare commit-gated
drafts; Flux uses EagerEndOfTurn and cancels them on TurnResumed. Metrics are
tagged per mode so the A/B is visible side by side. The measured low-latency
custom endpointer is the default; Flux remains an explicit A/B option.

The YAML plan contains evidence objectives, time/turn limits, and prohibited
topics—not interview question text. The LLM responds to the caller's intent and
uses those objectives as coverage goals rather than a questionnaire; it owns
wording, order, clarifications, and contextual follow-ups. Validated
`record_answer` and `end_call` tools run concurrently as provisional extraction,
so they never delay caller-facing speech while evidence and lifecycle state stay
backend-owned. Explicit stop requests
bypass ordinary dialogue generation; a narrow ASR-confusion path asks for
confirmation rather than continuing the interview or hanging up speculatively.

## Metrics that matter (check /metrics/{call_id} after a call)
endpoint_delay, llm_ttft (first text delta), tts_ttfb (first provider audio),
first_audio, turn_latency (p50/p95), prompt-cache tokens, barge-ins, turns.
Measured results and their history: see `REHEARSAL.md`.

## Verification
`python -m pytest -q` — 265 offline tests (no vendor calls; the browser capture
worklet runs as real JS under Node, since no Python test can reach it).
`python scripts/simulate_caller.py [--flux]` — live end-to-end gate: a synthesized
rambling caller with mid-thought pauses asserts turn integrity, extraction, and
that agent audio actually arrived (a silent TTS failure is a FAIL, not a pass).
`--protocol-self-test` runs its offline wire check.
