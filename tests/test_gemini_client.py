from types import SimpleNamespace

import pytest

from ai.gemini_client import GeminiClient, GeminiError


class _Models:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.models_used = []

    def generate_content(self, model, contents, config):
        self.calls += 1
        self.models_used.append(model)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def _client(cfg, tmp_path, replies, monkeypatch, key="test-key", model="gemini-test", fallbacks=()):
    monkeypatch.setenv("GEMINI_API_KEY", key)
    gcfg = cfg.gemini.model_copy(update={"model": model, "retry_backoff_seconds": 0, "fallback_models": fallbacks})
    c = GeminiClient(cfg.model_copy(update={"gemini": gcfg}), cache_dir=tmp_path)
    c._key = key  # load_dotenv must not override the test value
    models = _Models(replies)
    c._client = SimpleNamespace(models=models)
    return c, models


def test_retries_then_succeeds_and_caches(cfg, tmp_path, monkeypatch):
    c, models = _client(cfg, tmp_path, [TimeoutError(), "merhaba"], monkeypatch)
    assert c.generate_text("p", "s") == "merhaba"
    assert models.calls == 2
    assert c.generate_text("p", "s") == "merhaba"  # same prompt -> cache, no new call
    assert models.calls == 2


def test_gives_up_after_max_retries(cfg, tmp_path, monkeypatch):
    n = cfg.gemini.max_retries + 1
    c, models = _client(cfg, tmp_path, [TimeoutError()] * n, monkeypatch)
    with pytest.raises(GeminiError):
        c.generate_text("p", "s")
    assert models.calls == n


def test_missing_key_or_model_is_an_error(cfg, tmp_path, monkeypatch):
    c, _ = _client(cfg, tmp_path, ["x"], monkeypatch, key="")
    c._client = None
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        c.generate_text("p", "s")
    c2, _ = _client(cfg, tmp_path, ["x"], monkeypatch, model="")
    c2._client = None
    with pytest.raises(GeminiError, match="gemini.model"):
        c2.generate_text("p2", "s")


def test_key_never_logged(cfg, tmp_path, monkeypatch, caplog):
    secret = "super-secret-key-123"
    c, _ = _client(cfg, tmp_path, [TimeoutError(), "ok"], monkeypatch, key=secret)
    with caplog.at_level("DEBUG"):
        c.generate_text("p", "s")
    assert secret not in caplog.text


def test_falls_back_to_next_model(cfg, tmp_path, monkeypatch):
    n = cfg.gemini.max_retries + 1
    c, models = _client(cfg, tmp_path, [TimeoutError()] * n + ["yedek yanıt"], monkeypatch, fallbacks=["backup"])
    assert c.generate_text("p", "s") == "yedek yanıt"
    assert models.models_used == ["gemini-test"] * n + ["backup"]


def test_quota_error_skips_retries_and_uses_fallback(cfg, tmp_path, monkeypatch):
    quota = RuntimeError("429 RESOURCE_EXHAUSTED")
    quota.code = 429
    c, models = _client(cfg, tmp_path, [quota, "yedek"], monkeypatch, fallbacks=["backup"])
    assert c.generate_text("p", "s") == "yedek"
    assert models.models_used == ["gemini-test", "backup"]
