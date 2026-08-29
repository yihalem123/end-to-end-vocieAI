"""Validate the three vendor API keys with one cheap authenticated GET each.

## How this works
Each vendor exposes a free, read-only endpoint that fails with 401/403 on a bad key:
Deepgram lists projects, OpenAI lists models, ElevenLabs returns the account. We hit
each with its auth header style (Token / Bearer / xi-api-key) and report per-vendor
PASS/FAIL without ever printing the key itself (CLAUDE.md: no secrets in logs).
Sync httpx is fine here — this is a CLI script, not the async server. Exit code is
non-zero if any key fails, so it can gate the start of a build session.

Run from the repo root:  python scripts/check_keys.py
"""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.config import Settings  # noqa: E402

CHECKS = [
    # (vendor, url, header builder, settings attribute)
    ("Deepgram", "https://api.deepgram.com/v1/projects",
     lambda k: {"Authorization": f"Token {k}"}, "deepgram_api_key"),
    ("OpenAI", "https://api.openai.com/v1/models",
     lambda k: {"Authorization": f"Bearer {k}"}, "openai_api_key"),
    ("ElevenLabs", "https://api.elevenlabs.io/v1/user",
     lambda k: {"xi-api-key": k}, "elevenlabs_api_key"),
]


def main() -> int:
    settings = Settings()
    failures = 0
    for vendor, url, headers_for, attr in CHECKS:
        key = getattr(settings, attr)
        if not key:
            print(f"  MISSING  {vendor}: {attr.upper()} not set in .env")
            failures += 1
            continue
        try:
            resp = httpx.get(url, headers=headers_for(key), timeout=10)
        except httpx.HTTPError as exc:
            print(f"  ERROR    {vendor}: {type(exc).__name__} (network?)")
            failures += 1
            continue
        if resp.status_code == 200:
            print(f"  PASS     {vendor}")
        else:
            print(f"  FAIL     {vendor}: HTTP {resp.status_code}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
