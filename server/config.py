"""Env-based settings (pydantic-settings). Built in Phase 0.

## How this works
Settings is a pydantic-settings model: each field maps to an env var of the same name
(case-insensitive), with `.env` read as a fallback so local dev needs no shell exports.
Keys default to "" rather than raising, so the app boots without credentials and each
vendor client fails loudly only when actually used (and check_keys.py verifies them
up front). TURN_MODEL is config, not code, because the turn model is a measured
choice — Phase 4 benchmarks it and we may swap it without touching the engine.
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
    turn_model: str = "gpt-5.6-luna"
    flux_eot_threshold: float = 0.85  # higher = more patient with pauses
    flux_eager_eot_threshold: float = 0.6  # early draft; EndOfTurn still commits
    extract_model: str = "gpt-5.6-terra"  # post-call: quality over latency
    plan_path: str = "plans/icu_nurse.yaml"


def get_settings() -> Settings:
    return Settings()
