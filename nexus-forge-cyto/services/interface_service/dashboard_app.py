"""Nexus-Forge Cyto — صفحه اصلی داشبورد هندسی."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_SERVICES = Path(__file__).resolve().parents[1]
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from interface_service import dashboard_shared as ds  # noqa: E402
from interface_service import dashboard_ui as ui  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="Nexus-Forge Cyto",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ds.inject_lab_white_theme()
    ds.init_session_state()
    ds.render_sidebar_shell()
    if ds.render_cancellation_panel():
        st.stop()

    ui.render_hero(
        "Nexus-Forge Cyto",
        "خط لوله هندسی آفلاین — ورود تصویر یا مختصات، غنی‌سازی κ و مورفومتری",
    )
    ui.render_pipeline_stepper(ds.pipeline_active_step())
    st.markdown("")
    ui.render_status_cards()

    st.markdown("---")
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("شروع کار")
        st.markdown(
            """
            1. از منوی کناری **Geometry Core** را باز کنید.
            2. یکی از سه روش را انتخاب کنید:
               - **تصویر لام** (SAM 3 — نیاز به `ultralytics` + `sam3.pt`)
               - **مختصات** (XML / GeoJSON / JSON)
               - **فایل آماده** (INPUT_JSON)
            3. **اجرای هندسه** را بزنید و خروجی را دانلود کنید.
            """
        )
        if st.button("رفتن به Geometry Core →", type="primary"):
            st.switch_page("pages/1_Geometry_Core.py")

    with right:
        st.subheader("پیش‌نمایش آخرین خروجی")
        enriched = ds.resolve_enriched_json(ds.INPUT_JSON)
        ui.render_enriched_preview(enriched)
        if not enriched:
            st.info("هنوز خروجی غنی‌شده‌ای وجود ندارد.")

    st.caption(
        f"ریشه پروژه: `{ds.REPO_ROOT}` · "
        f"Rust: {'✓' if ds.rust_binary_built() else 'نیاز به cargo build --release'}"
    )


if __name__ == "__main__":
    main()
