"""Settings load from environment variables (Phase 0: keys + turn model)."""
import pytest

from server.config import Settings


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("TURN_MODEL", "some-other-model")
    settings = Settings(_env_file=None)  # env vars only; ignore any local .env
    assert settings.openai_api_key == "sk-test-123"
    assert settings.turn_model == "some-other-model"


def test_settings_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY", "TURN_MODEL"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.deepgram_api_key == ""
    assert settings.turn_model == "gpt-5.6-luna"
