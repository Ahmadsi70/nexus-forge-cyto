"""Nexus-Forge Cyto **Core** — lightweight single-page dashboard.

Pure-geometry cytology engine UI (no AI, no Cellpose, no GNN, no LLM).

Workflow:
  1. User uploads a cell-coordinate annotation (.xml / .geojson / .json / .csv)
  2. The Python adapter converts polygons → 32-vertex rings (segment JSON)
  3. The Rust CLI enriches the JSON with morphometrics + malignancy labels
  4. Results are shown as a table + a Malignant/Normal summary + download button

Run:
  streamlit run python/dashboard/app.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- Path bootstrap so coord_adapter resolves regardless of CWD. ------------
_DASHBOARD = Path(__file__).resolve().parent
_PYTHON = _DASHBOARD.parent  # python/
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

from coord_adapter import (  # noqa: E402
    segment_envelope_from_file,
    write_gold_segment_json,
)

# Locate the compiled Rust CLI relative to the repo root.
_REPO_ROOT = _PYTHON.parent
_CLI_BIN = _REPO_ROOT / "target" / "release" / "nexus-core-cli"
if os.name == "nt" and not _CLI_BIN.suffix:
    _CLI_BIN = _CLI_BIN.with_suffix(".exe")


# --------------------------------------------------------------------------- #
# Page chrome                                                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Nexus-Forge Cyto Core",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; }
    [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    .big-metric { font-size: 2.2rem; font-weight: 700; }
    .sub-metric { font-size: 1.1rem; color: #475569; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Nexus-Forge Cyto Core")
st.caption("Pure-geometry cytology engine · Deterministic morphometrics + malignancy scoring · No AI")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def ensure_cli() -> Path | None:
    """Return the CLI binary path if present, else warn and return None."""
    if _CLI_BIN.is_file():
        return _CLI_BIN
    st.warning(
        f"Compiled CLI not found at `{_CLI_BIN}`.\n\n"
        "Build it first with:\n"
        "```\ncargo build --release\n```"
    )
    return None


def run_enrichment(segment_json: Path, output_json: Path) -> tuple[bool, str]:
    """Invoke the Rust CLI in single-file mode. Returns (success, log)."""
    cli = ensure_cli()
    if cli is None:
        return False, "CLI binary missing"
    cmd = [str(cli), "single", "-i", str(segment_json), "-o", str(output_json)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(_REPO_ROOT)
        )
        log = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode == 0, log
    except subprocess.TimeoutExpired:
        return False, "Timed out after 300s"
    except Exception as exc:
        return False, str(exc)


def load_enriched(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Sidebar                                                                       #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Input formats")
    st.markdown(
        "- **JSON** — `{cell_count, cells:[[x,y]...]}`\n"
        "- **GeoJSON** — FeatureCollection of Polygons\n"
        "- **XML** — MoNuSeg / ImageScope `Region/Vertex`\n"
        "- **CSV** — `x,y` (optionally grouped by `polygon_id`)"
    )
    st.divider()
    st.markdown("### Engine")
    st.code(f"Binary: {_CLI_BIN.name}\nBuilt: {'✅' if _CLI_BIN.is_file() else '❌'}")
    st.caption("Deterministic · No network · No GPU · Pure Rust")


# --------------------------------------------------------------------------- #
# Main column                                                                   #
# --------------------------------------------------------------------------- #
upload_col, result_col = st.columns([1, 2], gap="large")

with upload_col:
    st.markdown("#### 1 · Upload cell coordinates")
    uploaded = st.file_uploader(
        "Annotation file",
        type=["json", "geojson", "xml", "csv"],
        key="core_upload",
        help="Expert or dataset-exported closed polygons around cell nuclei.",
    )
    run_btn = st.button("⚙️ Run Geometry Engine", type="primary", disabled=uploaded is None)

# Persist the last enriched result across reruns.
if "core_enriched_path" not in st.session_state:
    st.session_state["core_enriched_path"] = None

if run_btn and uploaded is not None:
    work = Path(tempfile.mkdtemp(prefix="nexus_core_"))
    in_file = work / uploaded.name
    in_file.write_bytes(uploaded.getvalue())

    segment_json = work / "segment.json"
    enriched_json = work / "enriched.json"

    with st.spinner("Normalizing polygons → 32-vertex rings…"):
        try:
            write_gold_segment_json(in_file, segment_json)
        except Exception as exc:
            st.error(f"Annotation parse failed: {exc}")
            st.stop()

    with st.spinner("Running Rust geometry engine…"):
        ok, log = run_enrichment(segment_json, enriched_json)

    if not ok:
        st.error("Geometry engine failed.")
        with st.expander("Log"):
            st.code(log)
        st.stop()

    st.session_state["core_enriched_path"] = str(enriched_json)
    st.session_state["core_log"] = log
    st.success("Enrichment complete.")
    st.rerun()


# --------------------------------------------------------------------------- #
# Results                                                                       #
# --------------------------------------------------------------------------- #
enriched_path = Path(st.session_state["core_enriched_path"]) if st.session_state.get("core_enriched_path") else None
data = load_enriched(enriched_path) if enriched_path else None

with result_col:
    if data is None:
        st.info("Awaiting input. Upload a coordinate file and press **Run Geometry Engine**.")
    else:
        clinical = data.get("clinical", [])
        sf = data.get("spatial_features", [])
        sa = data.get("spatial_analysis", {})
        n = len(clinical)
        n_mal = sum(1 for c in clinical if c == "Malignant")
        n_norm = n - n_mal
        pct_mal = (100.0 * n_mal / n) if n else 0.0

        st.markdown("#### 2 · Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total cells", str(n))
        m2.metric("Malignant", str(n_mal))
        m3.metric("Normal", str(n_norm))
        m4.metric("Malignant %", f"{pct_mal:.1f}%")

        with st.expander("Spatial analysis parameters"):
            st.json(sa)

        st.markdown("#### 3 · Per-cell morphometrics")
        if sf:
            df = pd.DataFrame(sf)
            # Friendly column order.
            front = [
                "cell_id", "area", "perimeter", "circularity", "eccentricity",
                "knn_density", "rbf_risk_score", "distance_to_tumor_edge",
            ]
            cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
            df = df[cols]
            df["clinical"] = clinical[: len(df)]
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Quick distribution plots (stdlib-only, no plotly needed).
            st.markdown("##### Area distribution (Malignant vs Normal)")
            try:
                mal_areas = [float(r["area"]) for r, c in zip(sf, clinical) if c == "Malignant"]
                norm_areas = [float(r["area"]) for r, c in zip(sf, clinical) if c == "Normal"]
                hist_df = pd.DataFrame({
                    "Malignant": pd.Series(mal_areas),
                    "Normal": pd.Series(norm_areas),
                })
                st.bar_chart(hist_df.sum(axis=1).rename("count").to_frame())  # placeholder safe
                st.caption(f"Malignant area mean: {np.mean(mal_areas):.1f} · Normal area mean: {np.mean(norm_areas):.1f}")
            except Exception:
                pass

        st.markdown("#### 4 · Download enriched JSON")
        st.download_button(
            "⬇️ enriched.json",
            data=json.dumps(data, indent=2).encode("utf-8"),
            file_name="enriched.json",
            mime="application/json",
        )

        with st.expander("Engine log"):
            st.code(st.session_state.get("core_log", ""))
