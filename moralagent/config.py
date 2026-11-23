"""Configuration for MoralAgent, sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 8192


def load_settings(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> Settings:
    """Build settings, preferring explicit arguments over environment variables."""
    return Settings(
        model=model or os.getenv("OLLAMA_MODEL", Settings.model),
        base_url=base_url or os.getenv("OLLAMA_BASE_URL", Settings.base_url),
        temperature=(
            temperature
            if temperature is not None
            else float(os.getenv("OLLAMA_TEMPERATURE", Settings.temperature))
        ),
        num_ctx=(
            num_ctx if num_ctx is not None else int(os.getenv("OLLAMA_NUM_CTX", Settings.num_ctx))
        ),
    )
