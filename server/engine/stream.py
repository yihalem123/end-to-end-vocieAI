"""Responses-API stream assembly and sentence chunking.

## How this works
The Responses API streams typed SSE events, and these two classes are the
whole lesson of that protocol:
  response.output_text.delta                -> spoken text
  response.output_item.added                -> a function_call item opens
  response.function_call_arguments.delta/done -> its JSON args accumulate/finish
StreamAssembler folds that event soup into text deltas plus completed
ToolCalls, and raises EngineStreamError for failures that arrive INSIDE an
HTTP 200 stream (found live: credit_balance_exhausted came back as an event).
SentenceChunker turns text deltas into complete sentences for TTS, with guards
for abbreviations, decimals and dotted acronyms - "Which U.S. state" must not
be cut into two utterances. A trailing terminal is held until flush() because
"Dr." may still be awaiting "Smith".
"""
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

_ABBREVIATIONS = ("dr.", "st.", "mr.", "mrs.", "ms.", "e.g.", "i.e.", "vs.", "etc.")
# Dotted acronyms/initialisms: "U.S.", "a.m.", "R.N.". Free-form model text uses
# these constantly where the old scripted plan text never did, and splitting on
# the final dot synthesizes a fragment ("Which U.S.") as its own utterance.
# Merging two sentences is a far cheaper error than chopping one in half.
_DOTTED_ACRONYM = re.compile(r"^(?:[a-z]\.){2,}$")
_TERMINAL = ".!?"


class SentenceChunker:
    def __init__(self) -> None:
        self._buf = ""

    def push(self, delta: str) -> Iterator[str]:
        self._buf += delta
        while True:
            cut = self._find_boundary()
            if cut is None:
                return
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                yield sentence

    def _find_boundary(self) -> int | None:
        for i, ch in enumerate(self._buf):
            if ch not in _TERMINAL:
                continue
            after = self._buf[i + 1: i + 2]
            if after and not after.isspace():
                continue  # mid-token: decimals like "4.5", or no space yet
            if not after:
                return None  # boundary may still be an abbreviation; wait for more
            head = self._buf[: i + 1]
            last_word = head.rsplit(None, 1)[-1].lower() if head.split() else ""
            if last_word in _ABBREVIATIONS or _DOTTED_ACRONYM.match(last_word):
                continue
            return i + 1
        return None

    def flush(self) -> str | None:
        tail = self._buf.strip()
        self._buf = ""
        return tail or None


class EngineStreamError(RuntimeError):
    """The Responses stream reported a failure event (arrives inside HTTP 200)."""


class EngineTimeout(EngineStreamError):
    """The provider exceeded a stage timeout (connect or inter-chunk read)."""


@dataclass
class ToolCall:
    name: str
    call_id: str
    arguments: dict | None  # None = args were malformed JSON


class StreamAssembler:
    def __init__(self) -> None:
        self.tool_calls: list[ToolCall] = []
        self._open: dict[str, dict] = {}  # item_id -> {name, call_id, buf}
        self.usage: dict = {}

    def feed(self, event: dict) -> list[str]:
        """Consume one typed event; return any text deltas it carried."""
        match event.get("type"):
            case "error":
                err = event.get("error", {})
                raise EngineStreamError(err.get("message", "stream error"))
            case "response.failed" | "response.incomplete":
                err = event.get("response", {}).get("error") or {}
                raise EngineStreamError(err.get("message", "response failed"))
            case "response.completed":
                self.usage = event.get("response", {}).get("usage") or {}
            case "response.output_text.delta":
                return [event.get("delta", "")]
            case "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    self._open[item["id"]] = {"name": item.get("name", ""),
                                              "call_id": item.get("call_id", ""),
                                              "buf": item.get("arguments", "")}
            case "response.function_call_arguments.delta":
                entry = self._open.get(event.get("item_id", ""))
                if entry is not None:
                    entry["buf"] += event.get("delta", "")
            case "response.function_call_arguments.done":
                entry = self._open.pop(event.get("item_id", ""), None)
                if entry is not None:
                    raw = event.get("arguments") or entry["buf"]
                    try:
                        args: dict | None = json.loads(raw)
                    except json.JSONDecodeError:
                        args = None
                    self.tool_calls.append(ToolCall(name=entry["name"],
                                                    call_id=entry["call_id"],
                                                    arguments=args))
        return []
