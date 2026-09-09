"""Reusable Streamlit UI blocks for the Nexus-Forge geometry dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from interface_service import dashboard_shared as ds


def render_hero(title: str, subtitle: str) -> None:
    """Clinical header strip."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {ds.LAB_SURFACE} 0%, #eef2ff 100%);
            border: 1px solid {ds.LAB_BORDER};
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        ">
            <h1 style="margin:0;font-size:1.75rem;color:{ds.LAB_TEXT};">{title}</h1>
            <p style="margin:0.35rem 0 0;color:{ds.LAB_MUTED};font-size:1rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_stepper(active: int = 1) -> None:
    """Horizontal 3-step workflow indicator."""
    steps = [
        (1, "دریافت داده", "تصویر یا مختصات"),
        (2, "هندسه و κ", "موتور Rust"),
        (3, "خروجی", "JSON غنی‌شده"),
    ]
    cols = st.columns(3)
    for col, (num, label, hint) in zip(cols, steps):
        done = num < active
        current = num == active
        bg = "#dcfce7" if done else ("#dbeafe" if current else ds.LAB_SURFACE)
        border = "#16a34a" if done else (ds.LAB_ACCENT if current else ds.LAB_BORDER)
        icon = "✓" if done else str(num)
        with col:
            st.markdown(
                f"""
                <div style="
                    text-align:center;padding:12px 8px;border-radius:10px;
                    background:{bg};border:2px solid {border};
                ">
                    <div style="font-size:1.25rem;font-weight:700;">{icon}</div>
                    <div style="font-weight:600;font-size:0.9rem;">{label}</div>
                    <div style="font-size:0.75rem;color:{ds.LAB_MUTED};">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_status_cards() -> None:
    """Artifact readiness summary aligned with pipeline gate semantics."""
    enriched = ds.resolve_enriched_json(ds.INPUT_JSON)
    input_ok = ds.INPUT_JSON.is_file()
    rust_ok = enriched is not None
    if ds.STAGE_V2_JSON.is_file():
        export_ok = ds.ds_fresh_export()
        export_label = "آماده" if export_ok else "منقضی"
        export_color = "#166534" if export_ok else "#b45309"
        export_bg = "#f0fdf4" if export_ok else "#fffbeb"
    else:
        export_ok = False
        export_label = "در انتظار"
        export_color = "#64748b"
        export_bg = ds.LAB_SURFACE

    cards = [
        ("ورودی", input_ok, ds.INPUT_JSON.name if input_ok else "—", None),
        ("غنی‌شده Rust", rust_ok, enriched.name if enriched else "—", None),
        ("خروجی نهایی", export_ok or ds.STAGE_V2_JSON.is_file(), ds.STAGE_V2_JSON.name if ds.STAGE_V2_JSON.is_file() else "—", (export_label, export_color, export_bg)),
        ("باینری Rust", ds.rust_binary_built(), ds.rust_release_bin().name if ds.rust_binary_built() else "cargo run", None),
    ]
    cols = st.columns(4)
    for col, (title, ok, detail, override) in zip(cols, cards):
        if override:
            label, color, bg = override
        else:
            color = "#166534" if ok else "#64748b"
            bg = "#f0fdf4" if ok else ds.LAB_SURFACE
            label = "آماده" if ok else "در انتظار"
        with col:
            st.markdown(
                f"""
                <div class="status-pill" style="background:{bg};">
                    <div style="font-size:0.75rem;color:{ds.LAB_MUTED};">{title}</div>
                    <div style="font-weight:700;color:{color};">{label}</div>
                    <div style="font-size:0.7rem;color:{ds.LAB_MUTED};overflow:hidden;text-overflow:ellipsis;">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_enriched_preview(path: Path | None) -> None:
    """Compact clinical / morphometry preview from enriched JSON."""
    if path is None or not path.is_file():
        return
    import json

    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.warning("خواندن خروجی غنی‌شده ممکن نشد.")
        return

    clinical = data.get("clinical") or []
    mal = sum(1 for c in clinical if str(c).lower() == "malignant")
    norm = sum(1 for c in clinical if str(c).lower() == "normal")
    sf = data.get("spatial_features") or []
    areas = [float(r.get("area", 0)) for r in sf if isinstance(r, dict)]
    circ = [float(r.get("circularity", 0)) for r in sf if isinstance(r, dict)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("تعداد سلول", data.get("cell_count", len(data.get("cells") or [])))
    c2.metric("برچسب بدخیم", mal)
    c3.metric("برچسب طبیعی", norm)
    if areas:
        c4.metric("میانه مساحت (px²)", f"{sorted(areas)[len(areas) // 2]:.1f}")
    if circ:
        st.caption(f"میانه گردی مرز: {sorted(circ)[len(circ) // 2]:.3f}")
