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
    turn_model: str = "gpt-5.6-luna"


def get_settings() -> Settings:
    return Settings()
