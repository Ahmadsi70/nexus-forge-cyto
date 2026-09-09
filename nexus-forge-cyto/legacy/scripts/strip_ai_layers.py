#!/usr/bin/env python3
"""
Strip AI/ML layers from nexus-forge-cyto → geometry-only (TPI-ready) layout.

Why: Cellpose, GNN, LLM, and Ollama stacks are optional overlays; the Rust+Mojo
geometry core can run offline from vector/JSON inputs alone.

Usage:
  python scripts/strip_ai_layers.py              # dry-run (default)
  python scripts/strip_ai_layers.py --execute    # delete + patch
  python scripts/strip_ai_layers.py --execute --backup

Safety: always review ``cleanup_report.json`` after a run.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import tarfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Paths removed wholesale (relative to repo root) ----------------------------

DELETE_PATHS: tuple[str, ...] = (
    "services/inference_core",
    "services/ingest_service/cellpose_ingest.py",
    "services/interface_service/integrated_pipeline_reporter.py",
    "services/interface_service/qupath_linker.py",
    "services/interface_service/pages/1_Vision_Ingest.py",
    "services/interface_service/pages/3_GNN_Inference.py",
    "services/interface_service/pages/4_Clinical_Reports.py",
    "scripts/train_gnn.py",
    "requirements-cellpose.txt",
    "tests/test_cellpose_integration.py",
    "tools/sirat",
    "scripts/export_annotations.py",
)

# Discovery heuristics (reported; only deleted with --execute if under repo)
DISCOVER_GLOBS: tuple[str, ...] = (
    "**/*infer_gnn*",
    "**/*train_gnn*",
    "**/*cellpose*",
    "**/*ollama*",
    "**/nexus_forge_model.pth",
    "**/graph_data.pt",
)

DISCOVER_KEYWORDS: tuple[str, ...] = (
    "torch_geometric",
    "NexusGAT",
    "cellpose",
    "NEXUS_LLM",
    "hovernet_instances",
)

# Functions dropped from dashboard_shared.py (AI / GNN / Cellpose / Folium)
DASHBOARD_FUNCTIONS_TO_DROP: frozenset[str] = frozenset(
    {
        "write_cellpose_bootstrap",
        "build_cellpose_cmd",
        "build_gnn_cmd",
        "build_qupath_cmd",
        "load_metrics_json",
        "load_attention_incoming",
        "folium_polygon_specs",
        "build_folium_map",
        "render_evaluation_metrics",
        "apply_lab_plotly_layout",
        "clinical_color",
    }
)

GEOMETRY_PIXI_TOML = textwrap.dedent(
    """\
    [workspace]
    name = "nexus-forge-cyto"
    version = "0.1.0"
    description = "Nexus-Forge geometry-only pipeline (Python / Rust / Mojo)"
    channels = ["https://conda.modular.com/max/", "conda-forge"]
    platforms = ["linux-64"]

    [dependencies]
    python = "==3.10"
    streamlit = ">=1.32,<2.0"
    mojo = { version = "==0.26.1.0", channel = "https://conda.modular.com/max/" }
    rust = ">=1.75"

    [pypi-dependencies]
    python-dotenv = ">=1.0,<2.0"
    pandas = ">=2.0,<3.0"
    numpy = ">=1.24,<3.0"
    pyarrow = ">=14.0,<20.0"
    lxml = ">=5.0,<6.0"
    jsonschema = ">=4.20,<5.0"
    scipy = ">=1.11,<2.0"

    [tasks]
    streamlit = "streamlit run services/interface_service/dashboard_app.py --server.port 8501 --server.address 0.0.0.0"
    """
)

GEOMETRY_DASHBOARD_APP = textwrap.dedent(
    '''\
    """Nexus-Forge geometry stepper — offline vector/Rust/Mojo shell."""

    from __future__ import annotations

    import sys
    from pathlib import Path

    import streamlit as st

    _SERVICES = Path(__file__).resolve().parents[1]
    if str(_SERVICES) not in sys.path:
        sys.path.insert(0, str(_SERVICES))

    from interface_service.dashboard_shared import (  # noqa: E402
        inject_lab_white_theme,
        init_session_state,
        render_cancellation_panel,
        render_sidebar_shell,
    )


    def main() -> None:
        st.set_page_config(
            page_title="Nexus-Forge Geometry Engine",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        inject_lab_white_theme()
        init_session_state()
        render_sidebar_shell()
        render_cancellation_panel()

        st.title("Nexus-Forge Geometry Engine")
        st.caption("Offline geometry pipeline — upload vector/JSON, run Rust enrichment.")
        st.markdown(
            "| Step | Gate |"
            "\\n|------|------|"
            "\\n| **1 Geometry** | Requires cell geometry file (`INPUT_JSON`) |"
        )


    if __name__ == "__main__":
        main()
    '''
)

GEOMETRY_PAGE = textwrap.dedent(
    '''\
    """Step 1 — Geometry Core: vector ingress + Rust batch + headless schema export."""

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

    from interface_service import dashboard_shared as ds

    ds.require_gate(ds.STEP_GEOMETRY, "Geometry Core")
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
                st.success("Geometry core complete.")
            else:
                st.error("Schema export failed — see Execution Logs.")
        elif section == "rust":
            ds.append_execution_log("Geometry — Rust Batch", log)
            st.error("Rust geometry failed — see Execution Logs.")

    st.subheader("Step 1 · Geometry Core (Rust / Headless Runtime)")
    exec_col, export_col = st.columns([3, 2], gap="large")

    with exec_col:
        st.markdown("#### Vector Ingress")
        vector_upload = st.file_uploader(
            "Cell geometry (.xml / .geojson / .json)",
            type=["json", "geojson", "xml"],
            key="geometry_vector_upload",
        )
        bootstrap_upload = st.file_uploader(
            "Or pre-built INPUT_JSON",
            type=["json"],
            key="geometry_bootstrap_upload",
        )

        if vector_upload is not None:
            st.session_state["vector_upload"] = vector_upload
        if bootstrap_upload is not None:
            ds.write_input_json_shortcut(bootstrap_upload)

        if st.button("Execute Geometry Core (Rust → Schema)", type="primary"):
            if st.session_state.get("vector_upload") is not None:
                try:
                    ds.write_vector_bootstrap(st.session_state["vector_upload"])
                except (ValueError, JSONDecodeError, XMLSyntaxError) as exc:
                    st.error(f"Vector ingest failed: {exc}")
                    st.stop()
            if not ds.INPUT_JSON.is_file():
                st.error("Upload a vector file or INPUT_JSON.")
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
        st.markdown('<div class="export-console">', unsafe_allow_html=True)
        ds.render_export_console("geometry")
        st.markdown("</div>", unsafe_allow_html=True)
    '''
)


@dataclass
class CleanupReport:
    mode: str
    timestamp: str
    deleted: list[str] = field(default_factory=list)
    patched: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)
    skipped_missing: list[str] = field(default_factory=list)
    discovered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _FunctionDropper(ast.NodeTransformer):
    """Remove top-level function definitions by name."""

    def __init__(self, names: frozenset[str]) -> None:
        self._names = names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name in self._names:
            return None
        return self.generic_visit(node)


def _resolve(path: str) -> Path:
    return REPO_ROOT / path


def discover_ai_artifacts() -> list[str]:
    """Scan repo for additional AI-related paths (informational + optional delete)."""
    found: set[str] = set()
    for pattern in DISCOVER_GLOBS:
        for p in REPO_ROOT.glob(pattern):
            if p.is_file() and ".pixi" not in p.parts and "strip_ai_layers" not in p.name:
                found.add(str(p.relative_to(REPO_ROOT)))
    return sorted(found)


def scan_keyword_hits(limit: int = 40) -> list[str]:
    hits: list[str] = []
    for py in REPO_ROOT.rglob("*.py"):
        if any(part in {".pixi", "__pycache__", ".git"} for part in py.parts):
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(kw in text for kw in DISCOVER_KEYWORDS):
            hits.append(str(py.relative_to(REPO_ROOT)))
        if len(hits) >= limit:
            break
    return hits


def _backup_paths(paths: Iterable[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        for p in paths:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(REPO_ROOT)))


def _drop_functions(source: str, names: frozenset[str]) -> str:
    tree = ast.parse(source)
    tree = _FunctionDropper(names).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _patch_dashboard_shared(text: str) -> str:
    """Apply geometry-only edits to dashboard_shared.py."""
    try:
        text = _drop_functions(text, DASHBOARD_FUNCTIONS_TO_DROP)
    except SyntaxError as exc:
        raise RuntimeError(f"dashboard_shared.py parse failed: {exc}") from exc

    replacements: tuple[tuple[str, str], ...] = (
        ("import folium\n", ""),
        ("import plotly.graph_objects as go\n", ""),
        ("import pandas as pd\n", ""),
        (
            "CELLPOSE_SCRIPT = REPO_ROOT / \"services\" / \"ingest_service\" / \"cellpose_ingest.py\"\n",
            "",
        ),
        (
            "QUPATH_LINKER_SCRIPT = REPO_ROOT / \"services\" / \"interface_service\" / \"qupath_linker.py\"\n",
            "",
        ),
        (
            "INFER_SCRIPT = REPO_ROOT / \"services\" / \"inference_core\" / \"infer_gnn.py\"\n",
            "",
        ),
        ("ATTENTION_CSV = OUTPUT_DIR / \"attention_map.csv\"\n", ""),
        ("GRAPH_PT = OUTPUT_DIR / \"graph_data.pt\"\n", ""),
        (
            'CLASS_LABELS = ["Epithelial_Malignant", "Inflammatory", "Lymphocyte", "Stromal"]\n',
            "",
        ),
        (
            'LABEL_COLORS = {"Malignant": "#E53935", "Immune": "#FFB300", "Normal": "#43A047"}\n',
            "",
        ),
        ("STEP_VISION = 1\n", ""),
        ("STEP_GEOMETRY = 2\n", "STEP_GEOMETRY = 1\n"),
        ("STEP_GNN = 3\n", ""),
        ("STEP_CLINICAL = 4\n", ""),
    )
    for old, new in replacements:
        text = text.replace(old, new)

    # ast.unparse emits single-quoted paths — strip stale AI constants.
    for line_prefix in (
        "CELLPOSE_SCRIPT = ",
        "QUPATH_LINKER_SCRIPT = ",
        "INFER_SCRIPT = ",
        "ATTENTION_CSV = ",
        "GRAPH_PT = ",
        "CLASS_LABELS = ",
        "LABEL_COLORS = ",
    ):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith(line_prefix)
        )

    text = re.sub(
        r"def gate_unlocked\(step: int\) -> bool:.*?return False\n",
        textwrap.dedent(
            '''\
            def gate_unlocked(step: int) -> bool:
                """Geometry-only gate: unlock when INPUT_JSON exists (staleness-aware)."""
                if step == STEP_GEOMETRY:
                    return INPUT_JSON.is_file()
                return False

            '''
        ),
        text,
        count=1,
        flags=re.DOTALL,
    )

    text = re.sub(
        r'gates = \[.*?\]',
        'gates = [("1 Geometry", gate_unlocked(STEP_GEOMETRY))]',
        text,
        count=1,
        flags=re.DOTALL,
    )

    text = re.sub(
        r'st\.session_state\["attn_layer"\].*?\n',
        "",
        text,
    )
    text = re.sub(
        r'st\.session_state\.setdefault\("attn_layer".*?\n',
        "",
        text,
    )
    text = re.sub(
        r'st\.session_state\.setdefault\("stage_c_slide_path".*?\n',
        "",
        text,
    )
    text = re.sub(
        r"if \"stage_c_slide_input\" not in st\.session_state.*?\n",
        "",
        text,
    )

    text = text.replace(
        "Remove all stage artifacts and return to Step 1 (Vision Ingest).",
        "Remove all stage artifacts and return to a clean geometry ingest state.",
    )

    # Slim reset artifact list
    text = re.sub(
        r"RESET_PIPELINE_ARTIFACTS: tuple\[Path, \.\.\.\] = \(.*?\)",
        """RESET_PIPELINE_ARTIFACTS: tuple[Path, ...] = (
    INPUT_JSON,
    STAGE_V2_JSON,
    OUTPUT_DIR / "stage_v2_bootstrap.json",
    OUTPUT_DIR / "stage_v2_projection.json",
    OUTPUT_DIR / "stage_v2_stats.json",
    OUTPUT_DIR / "annotation_index_manifest.json",
)""",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Remove attention slider block in sidebar
    text = re.sub(
        r'st\.session_state\["fill_opacity"\] = st\.slider\(.*?\n\s*\)\n',
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'st\.session_state\["attn_layer"\] = st\.selectbox\(.*?\n\s*\)\n',
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )

    # Slim export console
    text = re.sub(
        r"qupath_geojson = CHECKPOINT_B_DIR.*?\n",
        "",
        text,
    )
    text = re.sub(
        r'\s*\("QuPath GeoJSON", qupath_geojson, "application/geo\+json"\),\n',
        "",
        text,
    )
    text = re.sub(
        r'\s*\("stage_v2_final\.json", STAGE_V2_JSON, "application/json"\),\n',
        "",
        text,
    )

    return text


def _patch_headless_runtime(text: str) -> str:
    """Skip GNN graph export and HoverNet projection in geometry-only mode."""
    if "GEOMETRY_ONLY_MODE" in text:
        return text
    marker = 'if __name__ == "__main__":'
    guard = textwrap.dedent(
        '''\

        # Geometry-only: skip HoverNet projection and PyG export (no torch/GNN).
        GEOMETRY_ONLY_MODE = True

        '''
    )
    if marker in text and guard not in text:
        text = text.replace(f"\n{marker}", f"{guard}\n{marker}", 1)

    text = text.replace(
        "    payload = _project(payload, p[\"hovernet_instances\"], max_distance_px=max_distance_px, iou_threshold=iou_threshold)\n",
        "    if not GEOMETRY_ONLY_MODE:\n"
        "        payload = _project(payload, p[\"hovernet_instances\"], max_distance_px=max_distance_px, iou_threshold=iou_threshold)\n",
    )
    text = text.replace(
        '    _save(p["proj"], payload)\n',
        "    if not GEOMETRY_ONLY_MODE:\n        _save(p[\"proj\"], payload)\n",
        1,
    )
    text = text.replace(
        "    payload = _pyg(payload, p[\"pt\"], knn_k=knn_k)\n",
        "    if not GEOMETRY_ONLY_MODE:\n        payload = _pyg(payload, p[\"pt\"], knn_k=knn_k)\n",
    )
    text = text.replace(
        '    _validate(payload, schema, "pyg")\n',
        "    if not GEOMETRY_ONLY_MODE:\n        _validate(payload, schema, \"pyg\")\n",
        1,
    )
    return text


def _patch_interface_init(text: str) -> str:
    return (
        '"""Module D — Interface Service: Streamlit geometry dashboard + checkpoint export."""\n'
    )


def _patch_ingest_init(text: str) -> str:
    return '"""Ingest Service — vector-only (AI cellpose layer removed)."""\n'


def delete_ai_paths(report: CleanupReport, execute: bool) -> None:
    for rel in DELETE_PATHS:
        path = _resolve(rel)
        if not path.exists():
            report.skipped_missing.append(rel)
            continue
        if execute:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        report.deleted.append(rel)


def apply_patches(report: CleanupReport, execute: bool) -> None:
    patch_plan: list[tuple[str, str | callable]] = [
        ("pixi.toml", lambda _: GEOMETRY_PIXI_TOML),
        (
            "services/interface_service/dashboard_app.py",
            lambda _: GEOMETRY_DASHBOARD_APP,
        ),
        (
            "services/interface_service/dashboard_shared.py",
            _patch_dashboard_shared,
        ),
        (
            "services/geometry_core/headless_pipeline_runtime.py",
            _patch_headless_runtime,
        ),
        ("services/interface_service/__init__.py", _patch_interface_init),
        ("services/ingest_service/__init__.py", _patch_ingest_init),
    ]

    for rel, transform in patch_plan:
        path = _resolve(rel)
        if not path.is_file():
            report.warnings.append(f"patch skipped (missing): {rel}")
            continue
        original = path.read_text(encoding="utf-8")
        updated = transform(original) if callable(transform) else transform
        if updated != original:
            if execute:
                path.write_text(updated, encoding="utf-8")
            report.patched.append(rel)


def reorganize_pages(report: CleanupReport, execute: bool) -> None:
    old_page = _resolve("services/interface_service/pages/2_Geometry_Core.py")
    new_page = _resolve("services/interface_service/pages/1_Geometry_Core.py")
    if old_page.is_file() and not new_page.is_file():
        if execute:
            new_page.write_text(GEOMETRY_PAGE, encoding="utf-8")
            old_page.unlink()
        report.renamed.append("pages/2_Geometry_Core.py -> pages/1_Geometry_Core.py")
    elif not new_page.is_file() and execute:
        new_page.write_text(GEOMETRY_PAGE.replace("etix", "etree"), encoding="utf-8")
        report.patched.append("services/interface_service/pages/1_Geometry_Core.py")


def write_report(report: CleanupReport) -> Path:
    out = REPO_ROOT / "cleanup_report.json"
    payload = {
        "mode": report.mode,
        "timestamp": report.timestamp,
        "deleted": report.deleted,
        "patched": report.patched,
        "renamed": report.renamed,
        "skipped_missing": report.skipped_missing,
        "discovered_extra": report.discovered,
        "keyword_scan_hits": scan_keyword_hits(),
        "warnings": report.warnings,
    }
    if report.mode == "execute":
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove AI layers; keep Rust/Mojo geometry core.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply deletions and patches (default: dry-run only)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Tar.gz backup of touched paths before --execute",
    )
    args = parser.parse_args()

    mode = "execute" if args.execute else "dry-run"
    report = CleanupReport(mode=mode, timestamp=datetime.now(timezone.utc).isoformat())
    report.discovered = discover_ai_artifacts()

    targets = [_resolve(p) for p in DELETE_PATHS if _resolve(p).exists()]
    targets += [_resolve(p) for p in (
        "pixi.toml",
        "services/interface_service/dashboard_app.py",
        "services/interface_service/dashboard_shared.py",
        "services/geometry_core/headless_pipeline_runtime.py",
    )]

    if args.execute and args.backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = REPO_ROOT / ".ai_backup" / f"pre_strip_{stamp}.tar.gz"
        _backup_paths(targets, backup)
        print(f"backup={backup}")

    delete_ai_paths(report, execute=args.execute)
    apply_patches(report, execute=args.execute)
    reorganize_pages(report, execute=args.execute)

    if args.execute:
        write_report(report)

    print(f"mode={mode}")
    print(f"deleted={len(report.deleted)} patched={len(report.patched)} renamed={len(report.renamed)}")
    for rel in report.deleted:
        print(f"  - {rel}")
    for rel in report.patched:
        print(f"  ~ {rel}")
    for rel in report.renamed:
        print(f"  -> {rel}")
    if report.warnings:
        print("warnings:")
        for w in report.warnings:
            print(f"  ! {w}")
    if not args.execute:
        print("\n[dry-run] Re-run with --execute --backup to apply.")
    else:
        print(f"\nreport={REPO_ROOT / 'cleanup_report.json'}")
        print("manual: review keyword_scan_hits; run pytest + cargo test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
