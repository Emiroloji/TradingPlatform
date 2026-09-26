"""Gemini API client: key from .env, timeout, retries, on-disk response cache (KURALLAR §6, §10).

Callers depend on the small `LLM` protocol so tests (and a future provider) can swap the backend.
The API key is read from the environment only and never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from pydantic import BaseModel

from config import Config

logger = logging.getLogger(__name__)


class GeminiError(RuntimeError):
    """Any failure to get a usable answer (missing key/model, network, quota, timeout)."""


class LLM(Protocol):
    def generate_json(self, prompt: str, system: str, schema: type[BaseModel]) -> str: ...

    def generate_text(self, prompt: str, system: str) -> str: ...


class GeminiClient:
    def __init__(self, cfg: Config, cache_dir: Path | None = None) -> None:
        load_dotenv(cfg.root / ".env")
        self._key = os.environ.get("GEMINI_API_KEY", "")
        self._cfg = cfg.gemini
        self._cache = cache_dir or cfg.storage_path / "cache" / "gemini"
        self._client = None

    def _connect(self):
        if not self._key:
            raise GeminiError("GEMINI_API_KEY is not set in .env")
        if not self._cfg.model:
            raise GeminiError("gemini.model is empty in config.yaml")
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self._key, http_options=types.HttpOptions(timeout=self._cfg.timeout_seconds * 1000)
            )
        return self._client

    # ------------------------------------------------------------------ cache
    def _cache_path(self, kind: str, prompt: str, system: str) -> Path:
        # model-independent: the same news analysed by a fallback model is not analysed again
        digest = hashlib.sha256(f"{kind}|{system}|{prompt}".encode()).hexdigest()
        return self._cache / f"{digest}.json"

    def _cached(self, path: Path) -> str | None:
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if datetime.now() - datetime.fromisoformat(entry["at"]) > timedelta(days=self._cfg.cache_days):
            return None
        return entry["text"]

    def _store(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"at": datetime.now().isoformat(), "text": text}, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ calls
    def _call(self, kind: str, prompt: str, system: str, schema: type[BaseModel] | None) -> str:
        path = self._cache_path(kind, prompt, system)
        if (hit := self._cached(path)) is not None:
            logger.info("Gemini cache hit (%s)", kind)
            return hit

        from google.genai import types

        client = self._connect()
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._cfg.temperature,
            response_mime_type="application/json" if schema else "text/plain",
            response_schema=schema,
        )
        last_error: Exception | None = None
        models = [self._cfg.model, *self._cfg.fallback_models]
        for model in models:
            for attempt in range(self._cfg.max_retries + 1):
                try:
                    response = client.models.generate_content(model=model, contents=prompt, config=config)
                    text = response.text or ""
                    if not text.strip():
                        raise GeminiError("empty response")
                    self._store(path, text)
                    return text
                except Exception as exc:  # network, quota, server errors: retry, then next model
                    last_error = exc
                    # API error messages carry status/reason, never the key
                    logger.warning("Gemini %s [%s] attempt %d failed: %s", kind, model, attempt + 1, str(exc)[:160])
                    if getattr(exc, "code", None) == 429:
                        break  # quota exhausted: retrying the same model only burns more quota
                    if attempt < self._cfg.max_retries:
                        time.sleep(self._cfg.retry_backoff_seconds * (attempt + 1))
        raise GeminiError(f"Gemini {kind} failed on {models}") from last_error

    def generate_json(self, prompt: str, system: str, schema: type[BaseModel]) -> str:
        return self._call("json", prompt, system, schema)

    def generate_text(self, prompt: str, system: str) -> str:
        return self._call("text", prompt, system, None)
