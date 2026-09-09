"""Environment-driven defaults for Ollama URL, model, and optional system prompt."""

from __future__ import annotations

import os

# سبک (~۱GB)، تمرکز ریاضی/علمی؛ مناسب RAM محدود. نصب: ollama pull qwen2-math:1.5b
DEFAULT_OLLAMA_MODEL = "qwen2-math:1.5b"


def resolve_ollama_url(cli_value: str | None) -> str:
    if cli_value and str(cli_value).strip():
        return str(cli_value).strip()
    return os.environ.get("SIRAT_OLLAMA_URL", "http://127.0.0.1:11434").strip()


def resolve_ollama_model(cli_value: str | None) -> str:
    if cli_value and str(cli_value).strip():
        return str(cli_value).strip()
    return os.environ.get("SIRAT_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def resolve_ollama_system(cli_value: str | None) -> str | None:
    if cli_value is not None:
        s = str(cli_value).strip()
        return s if s else None
    env = os.environ.get("SIRAT_OLLAMA_SYSTEM")
    if env is None:
        return None
    s = env.strip()
    return s if s else None


def default_ollama_url_for_ui() -> str:
    return os.environ.get("SIRAT_OLLAMA_URL", "http://127.0.0.1:11434").strip()


def default_ollama_model_for_ui() -> str:
    return os.environ.get("SIRAT_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def default_ollama_system_for_ui() -> str:
    return os.environ.get("SIRAT_OLLAMA_SYSTEM", "")


def env_five_pillar_enabled() -> bool:
    """SIRAT_OLLAMA_FIVE_PILLAR=0|false|no|off disables logprob+embed pipeline."""
    v = os.environ.get("SIRAT_OLLAMA_FIVE_PILLAR", "1").strip().lower()
    return v not in ("0", "false", "no", "off")
