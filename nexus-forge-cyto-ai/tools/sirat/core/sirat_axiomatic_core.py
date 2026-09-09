"""SIRAT 5-pillar core + biomedical pipeline (axioms, innovations, optional logprob/embed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sirat.core.sirat_innovation_engine import InnovationRecommender


@dataclass
class SiratCoreReport:
    total_tokens: int
    noor_purified: float
    noor_purified_pct: float
    hadid_index: float
    mizan_temperature: float
    mizan_entropy: float
    haqq_geodesic_dist: float | None
    haqq_within_threshold: bool | None
    cache_integrity: float


@dataclass
class BiomedicalGuardrailResult:
    accepted: bool
    violated_axioms: list[str] = field(default_factory=list)
    nufudh_override_applied: bool = False
    mizan_rejected: bool = False
    core_report: SiratCoreReport | None = None
    suggested_innovations: list[dict[str, Any]] = field(default_factory=list)


class SiratAxiomaticCore:
    def __init__(self) -> None:
        self._innov = InnovationRecommender()

    def format_report(self, rep: SiratCoreReport) -> str:
        lines = [
            f"tokens={rep.total_tokens}",
            f"Noor purified %={rep.noor_purified_pct:.2f}",
            f"Hadid index={rep.hadid_index:.4f}",
            f"Mizan T={rep.mizan_temperature:.4f} H={rep.mizan_entropy:.4f}",
            f"L_cache integrity={rep.cache_integrity:.4f}",
        ]
        if rep.haqq_geodesic_dist is not None:
            lines.append(f"Haqq geodesic dist={rep.haqq_geodesic_dist:.4f} within={rep.haqq_within_threshold}")
        return "\n".join(lines)

    def process(
        self,
        logprobs: list[float],
        *,
        v_output: np.ndarray | None = None,
        v_axiom: np.ndarray | None = None,
    ) -> SiratCoreReport:
        lp = np.asarray(logprobs, dtype=np.float64)
        n = int(lp.size)
        if n == 0:
            return SiratCoreReport(
                total_tokens=0,
                noor_purified=0.0,
                noor_purified_pct=0.0,
                hadid_index=0.5,
                mizan_temperature=0.1,
                mizan_entropy=0.0,
                haqq_geodesic_dist=None,
                haqq_within_threshold=None,
                cache_integrity=1.0,
            )

        mean = float(np.mean(lp))
        std = float(np.std(lp)) if n > 1 else 0.1
        noor_purified = float(np.sum(lp > -1.0))
        noor_purified_pct = 100.0 * noor_purified / n
        hadid_index = float(1.0 / (1.0 + np.exp(mean)))
        mizan_temperature = max(0.01, std + 0.1)
        exp_lp = np.exp(lp - np.max(lp))
        p = exp_lp / (exp_lp.sum() + 1e-12)
        mizan_entropy = float(-np.sum(p * np.log(p + 1e-12)))
        cache_integrity = float(np.clip(1.0 - mizan_entropy / 10.0, 0.0, 1.0))

        haqq_geodesic_dist: float | None = None
        haqq_within_threshold: bool | None = None
        if v_output is not None and v_axiom is not None:
            vo = np.asarray(v_output, dtype=np.float64).ravel()
            va = np.asarray(v_axiom, dtype=np.float64).ravel()
            denom = (np.linalg.norm(vo) * np.linalg.norm(va)) + 1e-12
            sim = float(np.dot(vo, va) / denom)
            haqq_geodesic_dist = float(1.0 - sim)
            haqq_within_threshold = haqq_geodesic_dist < 0.5

        return SiratCoreReport(
            total_tokens=n,
            noor_purified=noor_purified,
            noor_purified_pct=float(np.clip(noor_purified_pct, 0.0, 100.0)),
            hadid_index=float(np.clip(hadid_index, 0.0, 1.0)),
            mizan_temperature=mizan_temperature,
            mizan_entropy=max(0.0, mizan_entropy),
            haqq_geodesic_dist=haqq_geodesic_dist,
            haqq_within_threshold=haqq_within_threshold,
            cache_integrity=cache_integrity,
        )

    def _check_axioms(self, text: str) -> list[str]:
        t = text.lower()
        violated: list[str] = []
        if "100%" in t and "binding affinity" in t and "dissociation" in t:
            violated.append("AXIOM_ZAWJ_MAQALID")
        if "thalidomide" in t and ("pregnancy" in t or "safe" in t):
            violated.extend(["AXIOM_TAYY", "AXIOM_ALAQ"])
        if "vioxx" in t and ("metabolic" in t or "ignore" in t or "cardiovascular" in t):
            violated.append("AXIOM_ANKABUT_URWAH")
        # de-dup preserve order
        seen: set[str] = set()
        out: list[str] = []
        for a in violated:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def process_biomedical(
        self,
        text: str,
        *,
        logprobs: list[float] | None = None,
        v_output: np.ndarray | None = None,
        v_axiom: np.ndarray | None = None,
    ) -> BiomedicalGuardrailResult:
        violated = self._check_axioms(text)
        if violated:
            return BiomedicalGuardrailResult(
                accepted=False,
                violated_axioms=violated,
                suggested_innovations=[],
            )

        core_report: SiratCoreReport | None = None
        if logprobs is not None and len(logprobs) > 0:
            core_report = self.process(logprobs, v_output=v_output, v_axiom=v_axiom)

        innovations = self._innov.analyze_and_suggest(text)
        return BiomedicalGuardrailResult(
            accepted=True,
            violated_axioms=[],
            core_report=core_report,
            suggested_innovations=innovations,
        )
