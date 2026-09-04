"""Env-based settings (pydantic-settings).

## How this works
Settings is a pydantic-settings model: each field maps to an env var of the same name
(case-insensitive), with `.env` read as a fallback so local dev needs no shell exports.
Keys default to "" rather than raising, so the app boots without credentials and each
vendor client fails loudly only when actually used (and check_keys.py verifies them
up front). TURN_MODEL is config, not code, because the turn model is a measured
choice — it was benchmarked and can be swapped without touching the engine.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepgram_api_key: str = ""
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    # "Sarah", a premade voice available to free API accounts. (The classic
    # "Rachel" id is a library voice now — the API rejects it on free tiers.)
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    # TTS is swappable behind Speaker's synthesize() interface. "aura" rides
    # the Deepgram key (measured warm ttfb 349-384 ms vs ElevenLabs 472 ms);
    # "elevenlabs" needs its own key and quota.
    tts_provider: str = "elevenlabs"
    aura_model: str = "aura-2-thalia-en"
    turn_model: str = "gpt-5.6-luna"
    # Provider processing tier for the caller-facing request only. An opt-in,
    # measured three ways and NOT adopted: a tiny-prompt probe showed
    # 1294 -> 713 ms (10/10), but with the real ~1000-token prompt and history
    # the gain collapsed to 993 -> 872 ms median with identical means (7/12),
    # and live it was 1217 -> 1177 ms - inside the noise. Prefill dominates
    # this prompt; the tier only shortens queueing. It is billed at a premium,
    # so "default" is the default; set "priority" to trial it.
    openai_speech_service_tier: str = "default"
    flux_eot_threshold: float = 0.85  # higher = more patient with pauses
    flux_eager_eot_threshold: float = 0.6  # early draft; EndOfTurn still commits
    extract_model: str = "gpt-5.6-terra"  # post-call: quality over latency
    plan_path: str = "plans/icu_nurse.yaml"
    # Telephony (Twilio Media Streams). The stream socket needs a public wss://
    # origin - locally that is an ngrok/cloudflared tunnel to port 8080. When
    # the auth token is set, /twilio/voice rejects requests that do not carry
    # a valid X-Twilio-Signature.
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    public_base_url: str = ""  # e.g. https://abc123.ngrok.app


def get_settings() -> Settings:
    return Settings()
