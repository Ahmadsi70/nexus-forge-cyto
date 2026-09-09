"""HTTP client for local Ollama: /api/tags, /api/chat (optional logprobs), /api/embed."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaChatResult:
    message: str
    raw: dict[str, Any]
    token_logprobs: list[float]


def logprobs_from_chat_response(data: dict[str, Any]) -> list[float]:
    lp = data.get("logprobs")
    if not lp:
        return []
    out: list[float] = []
    for item in lp:
        if isinstance(item, dict) and "logprob" in item:
            out.append(float(item["logprob"]))
    return out


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def ollama_health(base_url: str, timeout_s: float = 5.0) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
        return True
    except OSError:
        return False


def ollama_chat(
    base_url: str,
    model: str,
    user_text: str,
    *,
    system: str | None = None,
    timeout_s: float = 120.0,
    return_logprobs: bool = False,
    top_logprobs: int = 0,
) -> OllamaChatResult:
    url = base_url.rstrip("/") + "/api/chat"
    messages: list[dict[str, str]] = []
    if system and str(system).strip():
        messages.append({"role": "system", "content": str(system).strip()})
    messages.append({"role": "user", "content": user_text})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if return_logprobs:
        payload["logprobs"] = True
        if top_logprobs is not None:
            payload["top_logprobs"] = int(top_logprobs)

    try:
        data = _post_json(url, payload, timeout_s)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except OSError:
            detail = str(e)
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e

    msg_obj = data.get("message") or {}
    content = (msg_obj.get("content") or "").strip()
    token_logprobs = logprobs_from_chat_response(data)
    if not token_logprobs and return_logprobs:
        token_logprobs = _token_logprobs_from_completion_probabilities(data)

    return OllamaChatResult(message=content, raw=data, token_logprobs=token_logprobs)


def _token_logprobs_from_completion_probabilities(data: dict[str, Any]) -> list[float]:
    cps = data.get("completion_probabilities")
    if not isinstance(cps, list):
        return []
    out: list[float] = []
    for block in cps:
        if not isinstance(block, dict):
            continue
        probs = block.get("probs")
        if not isinstance(probs, list) or not probs:
            continue
        top = probs[0]
        if isinstance(top, dict) and "prob" in top:
            p = float(top["prob"])
            p = max(1e-12, min(1.0, p))
            out.append(math.log(p))
    return out


def ollama_embed(
    base_url: str,
    model: str,
    text: str,
    timeout_s: float = 120.0,
) -> list[float]:
    url = base_url.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": text}
    try:
        data = _post_json(url, payload, timeout_s)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except OSError:
            detail = str(e)
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError("Ollama embed response missing 'embedding' list")
    return [float(x) for x in emb]
