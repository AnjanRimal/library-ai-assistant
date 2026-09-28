"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Model
    model: str = field(default_factory=lambda: os.environ.get("LIA_MODEL", "claude-haiku-4-5-20251001"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("LIA_MAX_TOKENS", "1024")))
    max_agent_steps: int = field(default_factory=lambda: int(os.environ.get("LIA_MAX_STEPS", "6")))

    # Pricing in USD per million tokens, used for cost tracking.
    # Defaults match Claude Haiku 4.5; override them if you change LIA_MODEL.
    input_price_per_mtok: float = field(default_factory=lambda: _float("LIA_INPUT_PRICE_PER_MTOK", 1.0))
    output_price_per_mtok: float = field(default_factory=lambda: _float("LIA_OUTPUT_PRICE_PER_MTOK", 5.0))

    # Catalog backend: "local" (bundled JSON) or "api" (the Spring Boot library-api)
    catalog_backend: str = field(default_factory=lambda: os.environ.get("LIA_CATALOG", "local"))
    library_api_url: str = field(
        default_factory=lambda: os.environ.get("LIA_LIBRARY_API_URL", "http://localhost:8080")
    )

    # Embeddings: "hashing" (offline, lexical) or "sentence-transformers" (semantic)
    embedder: str = field(default_factory=lambda: os.environ.get("LIA_EMBEDDER", "hashing"))

    # Guardrails
    max_question_chars: int = field(
        default_factory=lambda: int(os.environ.get("LIA_MAX_QUESTION_CHARS", "1000"))
    )


def get_settings() -> Settings:
    return Settings()
