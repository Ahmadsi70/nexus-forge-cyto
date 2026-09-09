"""
Spatial biomarker report generation for pharma and clinical trials.

Produces a slide-level summary with:
  - Nuclear Atypia Index (mean malignancy score)
  - Tissue Architecture Disruption Score (RBF entropy)
  - Immune Infiltration Index (kNN density histogram)
  - Tumor Heterogeneity Score (std of eccentricity)
  - Per-cell feature distributions (quartiles, histograms)

Output formats: JSON (default) or PDF report (when reportlab is available).
"""

from __future__ import annotations

import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ── Biomarker indices ─────────────────────────────────────────────────────────


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (matches numpy)."""
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    if n == 1:
        return arr[0]
    k = (p / 100.0) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return arr[lo]
    frac = k - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def _entropy(values: list[float], bins: int = 20) -> float:
    """Shannon entropy of a value distribution (histogram-based)."""
    if not values:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log(hist)))


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and standard deviation."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=np.float64)
    return (float(np.mean(arr)), float(np.std(arr)))


def _gini(values: list[float]) -> float:
    """Gini coefficient of a value distribution (0 = equality, 1 = inequality)."""
    if not values:
        return 0.0
    arr = np.sort(np.array(values, dtype=np.float64))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n)


# ── Biomarker computer ────────────────────────────────────────────────────────


def compute_biomarkers(
    enriched: dict[str, Any],
    *,
    slide_id: str | None = None,
    stain_type: str = "H&E",
    magnification: str = "40x",
    include_distributions: bool = True,
    include_cell_list: bool = False,
) -> dict[str, Any]:
    """
    Compute slide-level spatial biomarkers from enriched output.

    Args:
        enriched: Output of /v1/enrich (must contain clinical[], spatial_features[])
        slide_id: Optional slide identifier for report provenance
        stain_type: Stain description ("H&E", "IHC", "IF", etc.)
        magnification: Microscope magnification
        include_distributions: Include quartile distributions
        include_cell_list: Include per-cell feature vectors (large output)

    Returns:
        Biomarker report dict ready for JSON serialization
    """
    clinical = enriched.get("clinical", [])
    spatial = enriched.get("spatial_features", [])
    cells = enriched.get("cells", enriched.get("gold_segment", {}).get("cells", []))
    confidence = enriched.get("confidence", [])

    n = len(clinical)
    if n == 0:
        return _empty_report(slide_id, stain_type, magnification)

    # Extract per-cell metrics
    malignancy_scores = []
    areas = []
    eccentricities = []
    circularities = []
    knn_densities = []
    rbf_risks = []
    edge_dists = []
    perimeters = []
    confidences = []

    malignant_count = 0
    immune_count = 0
    normal_count = 0

    for i in range(n):
        label = str(clinical[i]).strip().lower() if i < len(clinical) else "normal"
        if label == "malignant":
            malignant_count += 1
        elif label == "immune":
            immune_count += 1
        else:
            normal_count += 1

        sf = spatial[i] if i < len(spatial) else {}
        areas.append(float(sf.get("area", 0)))
        eccentricities.append(float(sf.get("eccentricity", 0)))
        circularities.append(float(sf.get("circularity", 0)))
        knn_densities.append(float(sf.get("knn_density", 0)))
        rbf_risks.append(float(sf.get("rbf_risk_score", 0)))
        edge_dists.append(float(sf.get("distance_to_tumor_edge", 0)))
        perimeters.append(float(sf.get("perimeter", 0)))

        if i < len(confidence):
            confidences.append(float(confidence[i]))
        else:
            confidences.append(0.85)

    # ── Slide-level biomarker indices ──────────────────────────────────

    # Nuclear Atypia Index (NAI): inverse-circularity + eccentricity composite
    # Higher -> more atypical nuclei
    atypia_scores = []
    for c, e in zip(circularities, eccentricities):
        atypia_scores.append((1.0 - c) * 0.5 + e * 0.5)
    nai_mean, nai_std = _mean_std(atypia_scores)

    # Tissue Architecture Disruption Score: entropy of RBF risk distribution
    # Higher -> more disorganized tissue
    tads = float(_entropy(rbf_risks, bins=20))

    # Immune Infiltration Index: skew of kNN density distribution
    # Positive skew -> focal immune clusters
    if knn_densities:
        k_arr = np.array(knn_densities, dtype=np.float64)
        k_mean = float(np.mean(k_arr))
        k_std = float(np.std(k_arr)) if len(k_arr) > 1 else 1.0
        skew = float(np.mean(((k_arr - k_mean) / max(k_std, 1e-12)) ** 3)) if k_std > 1e-12 else 0.0
    eae:
        skew = 0.0

    # Tumor Heterogeneity Score: Gini coefficient of eccentricity distribution
    # Higher -> more heterogeneous nuclear shape
    ths = float(_gini(eccentricities))

    # ── Statistics per feature ────────────────────────────────────────

    features_stats = {
        "area_px2": _distribution_summary(areas, "Continuous"),
        "eccentricity": _distribution_summary(eccentricities, "Ratio"),
        "circularity": _distribution_summary(circularities, "Ratio"),
        "knn_density": _distribution_summary(knn_densities, "Continuous"),
        "rbf_risk_score": _distribution_summary(rbf_risks, "Probability"),
        "distance_to_tumor_edge": _distribution_summary(edge_dists, "Distance (px)"),
        "perimeter_px": _distribution_summary(perimeters, "Continuous"),
        "confidence": _distribution_summary(confidences, "Probability"),
        "atypia_index": _distribution_summary(atypia_scores, "Probability"),
    }

    # ── Class distribution ────────────────────────────────────────────

    class_distribution = {
        "malignant": {
            "count": malignant_count,
            "fraction": float(malignant_count / n) if n > 0 else 0.0,
        },
        "immune": {
            "count": immune_count,
            "fraction": float(immune_count / n) if n > 0 else 0.0,
        },
        "normal": {
            "count": normal_count,
            "fraction": float(normal_count / n) if n > 0 else 0.0,
        },
    }

    # ── Report assembly ───────────────────────────────────────────────

    report: dict[str, Any] = {
        "meta": {
            "generator": "nexus-forge-cyto-biomarker",
            "version": "0.2.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "slide_id": slide_id,
            "stain_type": stain_type,
            "magnification": magnification,
        },
        "summary": {
            "total_cells": n,
            "class_distribution": class_distribution,
            "biomarkers": {
                "nuclear_atypia_index": {
                    "value": float(nai_mean),
                    "std": float(nai_std),
                    "interpretation": _interpret_nai(nai_mean),
                    "units": "probability (0=normal, 1=atypical)",
                },
                "tissue_architecture_disruption": {
                    "value": float(tads),
                    "interpretation": _interpret_tads(tads),
                    "units": "Shannon entropy (nats)",
                },
                "immune_infiltration_skew": {
                    "value": float(skew),
                    "interpretation": _interpret_skew(skew),
                    "units": "skewness (positive = focal clusters)",
                },
                "tumor_heterogeneity_score": {
                    "value": float(ths),
                    "interpretation": _interpret_ths(ths),
                    "units": "Gini coefficient (0=homogeneous, 1=maximally heterogeneous)",
                },
            },
        },
        "features": features_stats,
    }

    if include_distributions:
        report["distributions"] = {
            name: {
                "quartiles": {
                    "q1": float(_percentile(vals, 25)),
                    "q2": float(_percentile(vals, 50)),
                    "q3": float(_percentile(vals, 75)),
                },
                "iqr": float(_percentile(vals, 75) - _percentile(vals, 25)),
                "min": float(min(vals)) if vals else 0.0,
                "max": float(max(vals)) if vals else 0.0,
            }
            for name, vals in {
                "area_px2": areas,
                "eccentricity": eccentricities,
                "circularity": circularities,
                "knn_density": knn_densities,
                "rbf_risk_score": rbf_risks,
                "distance_to_tumor_edge": edge_dists,
                "perimeter_px": perimeters,
                "confidence": confidences,
                "atypia_index": atypia_scores,
            }.items()
        }

    if include_cell_list:
        report["cells"] = [
            {
                "cell_id": i,
                "label": str(clinical[i]) if i < len(clinical) else "unknown",
                "confidence": float(confidences[i]) if i < len(confidences) else 0.85,
                "area_px2": float(areas[i]) if i < len(areas) else 0.0,
                "eccentricity": float(eccentricities[i]) if i < len(eccentricities) else 0.0,
                "circularity": float(circularities[i]) if i < len(circularities) else 0.0,
                "knn_density": float(knn_densities[i]) if i < len(knn_densities) else 0.0,
                "rbf_risk": float(rbf_risks[i]) if i < len(rbf_risks) else 0.0,
                "edge_distance": float(edge_dists[i]) if i < len(edge_dists) else 0.0,
                "atypia_index": float(atypia_scores[i]) if i < len(atypia_scores) else 0.0,
            }
            for i in range(n)
        ]

    return report


# ── Distribution helpers ──────────────────────────────────────────────────────


def _distribution_summary(values: list[float], dtype: str) -> dict:
    if not values:
        return {"count": 0, "dtype": dtype}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "dtype": dtype,
    }


# ── Clinical interpretation helpers ───────────────────────────────────────────


def _interpret_nai(value: float) -> str:
    if value < 0.2:
        return "Low atypia — predominantly uniform, round nuclei. Consistent with benign or well-differentiated tissue."
    elif value < 0.4:
        return "Moderate atypia — some nuclear irregularity. May indicate low-grade dysplasia."
    elif value < 0.6:
        return "Elevated atypia — significant pleomorphism. Consistent with moderate-grade neoplasia."
    else:
        return "High atypia — marked nuclear pleomorphism. Consistent with high-grade malignancy."


def _interpret_tads(value: float) -> str:
    if value < 1.5:
        return "Low disruption — preserved tissue architecture. Cells are spatially organized."
    elif value < 2.5:
        return "Moderate disruption — some loss of spatial organization. Emerging invasive patterns."
    elif value < 3.5:
        return "High disruption — significant architectural disorder. Invasive growth patterns present."
    else:
        return "Severe disruption — chaotic spatial organization. Diffuse infiltration or discohesive growth."


def _interpret_skew(value: float) -> str:
    if value < -0.5:
        return "Uniform immune distribution — diffuse, non-clustered infiltration."
    elif value < 0.5:
        return "Balanced immune distribution — no significant clustering detected."
    elif value < 1.5:
        return "Focal immune clusters — localized immune infiltration suggestive of tertiary lymphoid structures."
    else:
        return "Prominent immune clusters — dense focal infiltrates. Possible immune hotspots."


def _interpret_ths(value: float) -> str:
    if value < 0.2:
        return "Low heterogeneity — uniform nuclear shape across tissue. Monomorphic population."
    elif value < 0.4:
        return "Moderate heterogeneity — some variation in nuclear shape. Mixed cell population."
    elif value < 0.6:
        return "High heterogeneity — significant variation. Pleomorphic tumour with diverse clones."
    else:
        return "Extreme heterogeneity — highly variable nuclear morphology. Clonal diversity or treatment effect."


def _empty_report(slide_id: str | None, stain_type: str, magnification: str) -> dict:
    return {
        "meta": {
            "generator": "nexus-forge-cyto-biomarker",
            "version": "0.2.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "slide_id": slide_id,
            "stain_type": stain_type,
            "magnification": magnification,
        },
        "summary": {
            "total_cells": 0,
            "class_distribution": {},
            "biomarkers": {},
        },
        "features": {},
        "error": "No cells in enriched output",
    }


# ── PDF report generation ─────────────────────────────────────────────────────


def generate_pdf_report(
    biomarker_report: dict[str, Any],
    output_path: Path | None = None,
) -> bytes:
    """
    Generate a PDF biomarker report.

    Requires reportlab: pip install reportlab

    Args:
        biomarker_report: Output of compute_biomarkers()
        output_path: If provided, write PDF to this path

    Returns:
        PDF bytes
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF generation. Install it: pip install reportlab"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=16, spaceAfter=6 * mm
    )
    heading_style = ParagraphStyle(
        "Heading2", parent=styles["Heading2"], fontSize=13, spaceAfter=3 * mm
    )
    normal_style = ParagraphStyle(
        "Normal2", parent=styles["Normal"], fontSize=10
    )

    story = []

    # Title
    meta = biomarker_report.get("meta", {})
    story.append(Paragraph("Nexus-Forge Cyto — Biomarker Report", title_style))
    story.append(Paragraph(
        f"Slide: {meta.get('slide_id', 'N/A')} | "
        f"Stain: {meta.get('stain_type', 'N/A')} | "
        f"Mag: {meta.get('magnification', 'N/A')} | "
        f"Generated: {meta.get('timestamp_utc', 'N/A')}",
        normal_style,
    ))
    story.append(Spacer(1, 6 * mm))

    # Summary
    summary = biomarker_report.get("summary", {})
    story.append(Paragraph("Slide Summary", heading_style))
    story.append(Paragraph(
        f"Total cells: <b>{summary.get('total_cells', 0)}</b>",
        normal_style,
    ))

    # Class distribution table
    cd = summary.get("class_distribution", {})
    if cd:
        dist_data = [["Class", "Count", "Fraction"]]
        for cls_name in ["malignant", "immune", "normal"]:
            cls_data = cd.get(cls_name, {})
            dist_data.append([
                cls_name.capitalize(),
                str(cls_data.get("count", 0)),
                f"{cls_data.get('fraction', 0) * 100:.1f}%",
            ])
        dist_table = Table(dist_data, colWidths=[60, 60, 60])
        dist_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#ffcdd2")),
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fff9c4")),
            ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#c8e6c9")),
        ]))
        story.append(dist_table)

    story.append(Spacer(1, 4 * mm))

    # Biomarkers
    biomarkers = summary.get("biomarkers", {})
    story.append(Paragraph("Biomarker Indices", heading_style))

    for name, bm in biomarkers.items():
        title = name.replace("_", " ").title()
        value = bm.get("value", 0)
        interp = bm.get("interpretation", "")
        units = bm.get("units", "")
        story.append(Paragraph(
            f"<b>{title}:</b> {value:.4f} ({units})<br/>{interp}",
            normal_style,
        ))
        story.append(Spacer(1, 2 * mm))

    story.append(Spacer(1, 4 * mm))

    # Feature statistics table
    features = biomarker_report.get("features", {})
    if features:
        story.append(Paragraph("Feature Statistics", heading_style))
        feat_data = [["Feature", "Mean", "Std", "Median", "Min", "Max"]]
        for fname, fstat in features.items():
            if fstat.get("count", 0) == 0:
                continue
            feat_data.append([
                fname,
                f"{fstat.get('mean', 0):.3f}",
                f"{fstat.get('std', 0):.3f}",
                f"{fstat.get('median', 0):.3f}",
                f"{fstat.get('min', 0):.3f}",
                f"{fstat.get('max', 0):.3f}",
            ])
        if len(feat_data) > 1:
            feat_table = Table(feat_data, colWidths=[80, 50, 50, 50, 50, 50])
            feat_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ]))
            story.append(feat_table)

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)

    return pdf_bytes


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test with synthetic data
    synthetic = {
        "clinical": ["Malignant"] * 30 + ["Immune"] * 15 + ["Normal"] * 55,
        "spatial_features": [
            {
                "area": 200 + np.random.normal(0, 30),
                "eccentricity": 0.3 + np.random.normal(0, 0.1),
                "circularity": 0.7 + np.random.normal(0, 0.05),
                "knn_density": 0.02 + np.random.normal(0, 0.005),
                "rbf_risk_score": 0.5 + np.random.normal(0, 0.15),
                "distance_to_tumor_edge": np.random.uniform(0, 300),
                "perimeter": 60 + np.random.normal(0, 10),
            }
            for _ in range(100)
        ],
        "confidence": [np.random.uniform(0.7, 1.0) for _ in range(100)],
    }
    report = compute_biomarkers(synthetic, slide_id="TEST-001")
    print(json.dumps(report, indent=2, ensure_ascii=False))