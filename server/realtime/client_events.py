"""Browser control messages -> typed pipeline events.

## How this works
The browser sends two things over the call socket: binary audio frames and
small JSON control messages - playback acknowledgements that carry a
generation id (cleared / playback_started / playback_drained /
playback_overflow) and text-mode chat. parse_client_message() turns one
JSON text into exactly one typed event or None. It is strict about the
things that matter for correctness (a generation id must be a positive
integer, played samples are clamped at zero, chat text is trimmed) and
forgiving about everything else: malformed JSON, unknown types and bad
field types all yield None, because a confused client must never be able to
kill a live call. The typed events are what the event router matches on.
"""
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ClientCleared:
    generation_id: int
    played_samples: int


@dataclass(frozen=True)
class ClientPlaybackDrained:
    generation_id: int


@dataclass(frozen=True)
class ClientPlaybackStarted:
    generation_id: int


@dataclass(frozen=True)
class ClientPlaybackOverflow:
    generation_id: int
    played_samples: int


@dataclass(frozen=True)
class ClientChat:
    text: str


ClientEvent = (ClientCleared | ClientPlaybackDrained | ClientPlaybackStarted
               | ClientPlaybackOverflow | ClientChat)


def parse_client_message(text: str) -> ClientEvent | None:
    """One JSON control message -> one typed event, or None if unusable."""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(msg, dict):
        return None
    try:
        match msg.get("type"):
            case "cleared":
                generation = int(msg.get("generation_id", 0))
                if generation > 0:
                    return ClientCleared(
                        generation, max(0, int(msg.get("played_samples", 0))))
            case "playback_drained":
                generation = int(msg.get("generation_id", 0))
                if generation > 0:
                    return ClientPlaybackDrained(generation)
            case "playback_started":
                generation = int(msg.get("generation_id", 0))
                if generation > 0:
                    return ClientPlaybackStarted(generation)
            case "playback_overflow":
                generation = int(msg.get("generation_id", 0))
                if generation > 0:
                    return ClientPlaybackOverflow(
                        generation, max(0, int(msg.get("played_samples", 0))))
            case "chat":
                chat_text = str(msg.get("text", "")).strip()
                if chat_text:
                    return ClientChat(chat_text)
    except (TypeError, ValueError):
        return None  # malformed client control messages never kill the call
    return None
