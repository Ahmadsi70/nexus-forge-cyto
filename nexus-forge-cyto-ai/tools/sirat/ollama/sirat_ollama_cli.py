"""
CLI: Ollama chat + SIRAT (با logprob و embed برای ۵ ستونه، در صورت پشتیبانی Ollama).

Environment:
  SIRAT_OLLAMA_URL, SIRAT_OLLAMA_MODEL (default qwen2-math:1.5b), SIRAT_OLLAMA_SYSTEM
  SIRAT_OLLAMA_FIVE_PILLAR=0  →  غیرفعال کردن logprob+embed (مثل قبل)

Examples:
  python -m sirat.ollama.sirat_ollama_cli -p "Explain first-order elimination briefly."
  python -m sirat.ollama.sirat_ollama_cli -p "Hi" --no-five-pillar
  python -m sirat.ollama.sirat_ollama_cli --ping --json-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sirat.core.biomedical_guardrail import BiomedicalGuardrailResult
from sirat.ollama.ollama_bridge import ollama_health
from sirat.core.sirat_axiomatic_core import SiratAxiomaticCore
from sirat.ollama.sirat_ollama_pipeline import ollama_generate_for_sirat
from sirat.ollama.sirat_runtime_settings import (
    env_five_pillar_enabled,
    resolve_ollama_model,
    resolve_ollama_system,
    resolve_ollama_url,
)


def _sirat_dict(core: SiratAxiomaticCore, r: BiomedicalGuardrailResult) -> dict:
    out: dict = {
        "accepted": r.accepted,
        "violated_axioms": list(r.violated_axioms),
        "nufudh_override_applied": r.nufudh_override_applied,
        "mizan_rejected": r.mizan_rejected,
        "has_core_report": r.core_report is not None,
        "innovations": [
            {"id": i.get("id"), "domain": i.get("domain")} for i in r.suggested_innovations
        ],
    }
    if r.core_report is not None:
        rep = r.core_report
        out["core"] = {
            "total_tokens": rep.total_tokens,
            "noor_purified_pct": rep.noor_purified_pct,
            "hadid_index": rep.hadid_index,
            "mizan_temperature": rep.mizan_temperature,
            "mizan_entropy": rep.mizan_entropy,
            "haqq_geodesic_dist": rep.haqq_geodesic_dist,
            "haqq_within_threshold": rep.haqq_within_threshold,
            "cache_integrity": rep.cache_integrity,
        }
    return out


def _emit_json(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), file=sys.stderr)


def _print_sirat_human(core: SiratAxiomaticCore, r: BiomedicalGuardrailResult) -> None:
    print("\n--- SIRAT ---", file=sys.stderr)
    print("accepted:", r.accepted, file=sys.stderr)
    if r.violated_axioms:
        print("violated_axioms:", ", ".join(r.violated_axioms), file=sys.stderr)
    if r.nufudh_override_applied:
        print("nufudh_override_applied: True", file=sys.stderr)
    if r.mizan_rejected and not r.nufudh_override_applied:
        print("mizan_rejected: True", file=sys.stderr)
    if r.core_report is not None:
        print(core.format_report(r.core_report), file=sys.stderr)
    else:
        print("(no 5-pillar report)", file=sys.stderr)
    if r.accepted and r.suggested_innovations:
        print("\nInnovation suggestions:", file=sys.stderr)
        for inv in r.suggested_innovations:
            print(f"  - {inv.get('id')}: {inv.get('domain')}", file=sys.stderr)
            print(f"    {inv.get('innovation', '')}", file=sys.stderr)


def _run_sirat(
    core: SiratAxiomaticCore,
    text: str,
    *,
    json_summary: bool,
    model_response: str | None,
    logprobs: list[float] | None = None,
    v_output=None,
    v_axiom=None,
) -> int:
    r = core.process_biomedical(
        text.strip(),
        logprobs=logprobs,
        v_output=v_output,
        v_axiom=v_axiom,
    )
    if json_summary:
        payload: dict = {"sirat": _sirat_dict(core, r)}
        if model_response is not None:
            payload["model_response"] = model_response
        if logprobs is not None:
            payload["token_logprob_count"] = len(logprobs)
        _emit_json(payload)
    else:
        _print_sirat_human(core, r)
    return 0 if r.accepted else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Ollama + SIRAT biomedical guardrail CLI")
    p.add_argument("--prompt", "-p", help="User message to send to Ollama")
    p.add_argument("--system", "-s", default=None, help="System prompt (or SIRAT_OLLAMA_SYSTEM)")
    p.add_argument("--model", "-m", default=None, help="Model (or SIRAT_OLLAMA_MODEL)")
    p.add_argument("--url", "-u", default=None, help="Ollama URL (or SIRAT_OLLAMA_URL)")
    p.add_argument("--text-file", "-f", type=Path, help="Skip Ollama; validate UTF-8 file")
    p.add_argument("--stdin", action="store_true", help="Skip Ollama; read stdin")
    p.add_argument("--skip-sirat", action="store_true", help="Only print model output")
    p.add_argument("--ping", action="store_true", help="GET /api/tags")
    p.add_argument("--json-summary", action="store_true", help="JSON on stderr")
    p.add_argument(
        "--no-five-pillar",
        action="store_true",
        help="Disable logprob+embed (override SIRAT_OLLAMA_FIVE_PILLAR)",
    )
    args = p.parse_args()

    url = resolve_ollama_url(args.url)
    model = resolve_ollama_model(args.model)
    system = resolve_ollama_system(args.system)
    use_five = env_five_pillar_enabled() and not args.no_five_pillar

    if args.ping:
        ok = ollama_health(url)
        if args.json_summary:
            _emit_json({"ping": "ok" if ok else "unreachable", "url": url})
        else:
            print("ok" if ok else "unreachable", file=sys.stderr)
        return 0 if ok else 2

    core = SiratAxiomaticCore()

    if args.stdin:
        text = sys.stdin.read()
        if not text.strip():
            if args.json_summary:
                _emit_json({"error": "stdin_empty"})
            else:
                print("stdin empty", file=sys.stderr)
            return 2
        return _run_sirat(core, text, json_summary=args.json_summary, model_response=None)

    if args.text_file is not None:
        path = args.text_file
        if not path.is_file():
            if args.json_summary:
                _emit_json({"error": "not_a_file", "path": str(path)})
            else:
                print(f"not a file: {path}", file=sys.stderr)
            return 2
        text = path.read_text(encoding="utf-8", errors="replace")
        return _run_sirat(core, text, json_summary=args.json_summary, model_response=None)

    if not args.prompt or not args.prompt.strip():
        p.error("provide --prompt, or --text-file, or --stdin, or --ping")

    msg, lps, vo, va, emb_err = ollama_generate_for_sirat(
        url,
        model,
        args.prompt.strip(),
        system,
        use_five_pillar=use_five,
    )
    print(msg)
    if use_five and emb_err:
        print(f"(Haqq embed skipped: {emb_err})", file=sys.stderr)
    if use_five and not lps:
        print(
            "(no token logprobs in response; update Ollama or try another model)",
            file=sys.stderr,
        )

    if args.skip_sirat:
        if args.json_summary:
            pl: dict = {"model_response": msg, "sirat_skipped": True, "five_pillar_requested": use_five}
            if lps is not None:
                pl["token_logprob_count"] = len(lps)
            _emit_json(pl)
        return 0

    if not msg.strip():
        if args.json_summary:
            _emit_json({"error": "empty_model_response"})
        else:
            print("(empty model response; skipping SIRAT)", file=sys.stderr)
        return 2

    return _run_sirat(
        core,
        msg,
        json_summary=args.json_summary,
        model_response=msg,
        logprobs=lps,
        v_output=vo,
        v_axiom=va,
    )


if __name__ == "__main__":
    raise SystemExit(main())
