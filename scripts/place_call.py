"""Place an outbound screening call through Twilio's REST API.

The call is answered by Twilio, which fetches TwiML from PUBLIC_BASE_URL/twilio/voice
and connects the audio to wss://PUBLIC_BASE_URL/ws/twilio - the same pipeline the
browser uses, behind the Twilio adapter. Needs TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
TWILIO_FROM_NUMBER and PUBLIC_BASE_URL (a public tunnel to the local server).

  python scripts/place_call.py +15551234567
"""
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.config import Settings  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: place_call.py +E164_NUMBER", file=sys.stderr)
        return 2
    s = Settings()
    missing = [n for n in ("twilio_account_sid", "twilio_auth_token",
                           "twilio_from_number", "public_base_url") if not getattr(s, n)]
    if missing:
        print("missing settings: " + ", ".join(n.upper() for n in missing), file=sys.stderr)
        return 2
    resp = httpx.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{s.twilio_account_sid}/Calls.json",
        auth=(s.twilio_account_sid, s.twilio_auth_token),
        data={"To": argv[1], "From": s.twilio_from_number,
              "Url": s.public_base_url.rstrip("/") + "/twilio/voice"},
        timeout=20,
    )
    if resp.status_code >= 300:
        print(f"twilio {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
        return 1
    print("call queued:", resp.json().get("sid"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
