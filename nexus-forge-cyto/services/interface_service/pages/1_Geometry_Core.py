"""Geometry Core — دریافت تصویر/مختصات، اجرای Rust، خروجی JSON."""

from __future__ import annotations

import os
import sys
from json import JSONDecodeError
from pathlib import Path

import streamlit as st
from lxml.etree import XMLSyntaxError

_SERVICES = Path(__file__).resolve().parents[2]
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from interface_service import dashboard_shared as ds  # noqa: E402
from interface_service import dashboard_ui as ui  # noqa: E402

st.set_page_config(page_title="Geometry Core", layout="wide")
ds.inject_lab_white_theme()
ds.init_session_state()
ds.render_sidebar_shell()

if ds.render_cancellation_panel():
    st.stop()

code, log = ds.consume_last_subprocess_result("Geometry")
if code is not None:
    section = st.session_state.pop("_geometry_stage", "")
    if section == "rust" and code == 0:
        ds.append_execution_log("Geometry — Rust Batch", log)
        enriched = ds.resolve_enriched_json(ds.INPUT_JSON)
        if enriched:
            st.session_state["_geometry_stage"] = "graph"
            cmd, env = ds.build_graph_export_cmd(enriched)
            ds.launch_cancellable_subprocess(
                cmd,
                env=env,
                section="Geometry — Graph Export",
                partial_paths=[ds.STAGE_V2_JSON],
            )
            st.rerun()
    elif section == "graph":
        ds.append_execution_log("Geometry — Graph Export", log)
        if code == 0 and ds.STAGE_V2_JSON.is_file():
            st.success("✅ خط لوله هندسی با موفقیت تکمیل شد.")
        else:
            st.error("خطای اعتبارسنجی schema — Execution Logs را ببینید.")
    elif section == "rust":
        ds.append_execution_log("Geometry — Rust Batch", log)
        st.error("خطای موتور Rust — Execution Logs را ببینید.")

ui.render_hero("Geometry Core", "دریافت داده → κ و مورفومتری → خروجی نهایی")
ui.render_pipeline_stepper(ds.pipeline_active_step())
st.markdown("")
ui.render_status_cards()

enriched = ds.resolve_enriched_json(ds.INPUT_JSON)
if enriched:
    st.markdown("#### پیش‌نمایش نتایج")
    ui.render_enriched_preview(enriched)

st.markdown("---")
exec_col, export_col = st.columns([3, 2], gap="large")

with exec_col:
    tab_img, tab_vec, tab_ready = st.tabs(["📷 تصویر لام", "📐 مختصات", "📁 فایل آماده"])

    with tab_img:
        st.caption("تقسیم خودکار هسته با SAM 3 — subprocess جدا از UI")
        st.info("نیاز: `pip install ultralytics opencv-python-headless` و `NEXUS_SAM3_MODEL_PATH`")
        image_upload = st.file_uploader(
            "PNG / JPG / TIFF",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="geometry_image_upload",
        )
        if image_upload is not None:
            st.session_state["image_upload"] = image_upload
            st.image(image_upload, caption="پیش‌نمایش", width="stretch")

    with tab_vec:
        st.caption("XML (ImageScope/MoNuSeg)، GeoJSON (QuPath)، JSON")
        vector_upload = st.file_uploader(
            "فایل مختصات",
            type=["json", "geojson", "xml"],
            key="geometry_vector_upload",
        )
        if vector_upload is not None:
            st.session_state["vector_upload"] = vector_upload
            st.success(f"بارگذاری شد: {vector_upload.name}")

    with tab_ready:
        bootstrap_upload = st.file_uploader(
            "INPUT_JSON از قبل ساخته‌شده",
            type=["json"],
            key="geometry_bootstrap_upload",
        )
        if bootstrap_upload is not None:
            ds.write_input_json_shortcut(bootstrap_upload)
            st.success("فایل آماده پذیرفته شد.")

    st.markdown("")
    run_col, clear_col = st.columns([2, 1])
    with run_col:
        run_clicked = st.button("▶ اجرای هندسه (Rust → Schema)", type="primary", width="stretch")
    with clear_col:
        if st.button("پاک‌سازی ورودی", width="stretch"):
            ds.reset_pipeline(hard=False)
            st.rerun()

    if run_clicked:
        try:
            with st.spinner("در حال پردازش ورودی… (SAM 3 ممکن است چند دقیقه طول بکشد)"):
                if st.session_state.get("image_upload") is not None:
                    ds.write_image_bootstrap(st.session_state["image_upload"])
                elif st.session_state.get("vector_upload") is not None:
                    ds.write_vector_bootstrap(st.session_state["vector_upload"])
        except (ValueError, JSONDecodeError, XMLSyntaxError) as exc:
            st.error(f"خطای دریافت داده: {exc}")
            st.stop()

        if not ds.INPUT_JSON.is_file():
            st.error("لطفاً تصویر، مختصات یا فایل JSON آپلود کنید.")
        else:
            st.session_state["_geometry_stage"] = "rust"
            cmd = ds.build_rust_geometry_cmd(ds.INPUT_JSON)
            ds.launch_cancellable_subprocess(
                cmd,
                env=dict(os.environ),
                section="Geometry — Rust Batch",
                partial_paths=[ds.RUST_OUT_DIR, ds.STAGE_V2_JSON],
            )
            st.rerun()

with export_col:
    st.markdown("#### دانلود خروجی")
    st.markdown('<div class="export-console">', unsafe_allow_html=True)
    ds.render_export_console("geometry")
    st.markdown("</div>", unsafe_allow_html=True)
