"""
Shared Streamlit shell: Lab White theme, artifact gates, cancellable subprocess engine.
Why: isolate multipage stepper UI from stage logic while preserving M-01 purge and vector precedence.
"""
from __future__ import annotations
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICES = REPO_ROOT / 'services'
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))
from api_service.pipeline_core import build_headless_export_cmd, gold_segment_from_vector_bytes
from nexus_core.runtime_paths import is_windows_host, production_output_dir, python_executable
from nexus_core.vector_ingest import FORMAT_JSON_STAGE, FORMAT_SAM3_NUCLEI, detect_vector_format
from interface_service.checkpoint_serializers import (
    read_file_bytes_if_exists,
    write_qupath_from_enriched,
)
# Prefer repo-local .env; never require parent workspace secrets for the dashboard.
_repo_env = REPO_ROOT / '.env'
if _repo_env.is_file():
    load_dotenv(_repo_env)
else:
    load_dotenv(REPO_ROOT.parent / '.env')
logger = logging.getLogger(__name__)
OUTPUT_DIR = production_output_dir()
INGEST_DIR = OUTPUT_DIR / 'dashboard_ingest'
RUST_OUT_DIR = OUTPUT_DIR / 'dashboard_rust_enriched'
INPUT_JSON = INGEST_DIR / 'uploaded_tissue.json'
STAGE_V2_JSON = OUTPUT_DIR / 'stage_v2_final.json'
CHECKPOINT_B_DIR = OUTPUT_DIR / 'checkpoint_b'
CHECKPOINT_C_DIR = OUTPUT_DIR / 'checkpoint_c'
QUPATH_GEOJSON = CHECKPOINT_B_DIR / 'qupath_annotations.geojson'
SCHEMA_PATH = REPO_ROOT / 'schemas' / 'enriched_graph_v2.schema.json'
RUST_RELEASE_BIN = REPO_ROOT / 'target' / 'release' / 'nexus-forge-cyto-batch'

def rust_release_bin() -> Path:
    """Resolved release binary path (`.exe` on Windows)."""
    if is_windows_host():
        exe = RUST_RELEASE_BIN.with_suffix('.exe')
        return exe if exe.is_file() else RUST_RELEASE_BIN
    return RUST_RELEASE_BIN
DEFAULT_FILL_OPACITY = 0.45
MAX_EXEC_LOG_LINES = 1000
MAX_LOG_LINE_BYTES = 5000
LOG_TRUNCATION_SUFFIX = '... [Truncated for Memory Safety]'
LAB_BG = '#ffffff'
LAB_SURFACE = '#f8fafc'
LAB_TEXT = '#0f172a'
LAB_MUTED = '#475569'
LAB_BORDER = '#e2e8f0'
LAB_ACCENT = '#1e40af'
LAB_ACCENT_HOVER = '#1d4ed8'
STEP_GEOMETRY = 1

def inject_lab_white_theme() -> None:
    """Apply pure Laboratory White clinical chrome globally."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {LAB_BG}; color: {LAB_TEXT}; }}
        [data-testid="stAppViewContainer"] {{ background: {LAB_BG}; }}
        [data-testid="stSidebar"] {{ background: {LAB_SURFACE}; border-right: 1px solid {LAB_BORDER}; }}
        h1,h2,h3,h4,h5,h6,p {{ color: {LAB_TEXT} !important; }}
        .stButton>button,.stDownloadButton>button {{
            background:{LAB_ACCENT}!important;color:#fff!important;border:1px solid {LAB_ACCENT}!important;
            border-radius:8px!important;font-weight:600!important;
        }}
        .stButton>button:hover,.stDownloadButton>button:hover {{
            background:{LAB_ACCENT_HOVER}!important;border-color:{LAB_ACCENT_HOVER}!important;
        }}
        [data-testid="stFileUploader"] section {{
            background:{LAB_SURFACE};border:1px dashed {LAB_BORDER};border-radius:10px;padding:8px;
        }}
        [data-testid="stTabs"] button {{
            font-weight:600;border-radius:8px 8px 0 0;
        }}
        .export-console {{
            background:{LAB_SURFACE};border:1px solid {LAB_BORDER};border-radius:12px;padding:16px 18px;
        }}
        .status-pill {{
            padding:10px 14px;border-radius:8px;border:1px solid {LAB_BORDER};background:{LAB_SURFACE};
            min-height:72px;
        }}
        .nav-lock {{
            background:#fef2f2;border:1px solid #fecaca;color:#991b1b;padding:10px;border-radius:8px;
            font-weight:600;margin-bottom:12px;
        }}
        div[data-testid="stMetric"] {{
            background:{LAB_SURFACE};border:1px solid {LAB_BORDER};border-radius:8px;padding:8px 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def init_session_state() -> None:
    """Bootstrap session keys for stepper, logs, and cancellable jobs."""
    for path in (OUTPUT_DIR, INGEST_DIR, RUST_OUT_DIR, CHECKPOINT_B_DIR, CHECKPOINT_C_DIR):
        path.mkdir(parents=True, exist_ok=True)
    st.session_state.setdefault('execution_log', 'No runs yet.')
    st.session_state.setdefault('is_processing', False)
    st.session_state.setdefault('active_proc', None)
    st.session_state.setdefault('active_proc_log', [])
    st.session_state.setdefault('active_proc_section', '')
    st.session_state.setdefault('active_proc_partials', [])
    st.session_state.setdefault('fill_opacity', DEFAULT_FILL_OPACITY)
RESET_PIPELINE_ARTIFACTS: tuple[Path, ...] = (
    INPUT_JSON,
    STAGE_V2_JSON,
    OUTPUT_DIR / "stage_v2_bootstrap.json",
    OUTPUT_DIR / "stage_v2_projection.json",
    OUTPUT_DIR / "stage_v2_stats.json",
    OUTPUT_DIR / "annotation_index_manifest.json",
)

def reset_pipeline(*, hard: bool=False) -> int:
    """Wipe pipeline artifacts so the stepper returns to a clean Vision-Ingest state.

    - Clears Streamlit session state keys that reference prior runs (uploads,
      stage flags, last subprocess result, execution log) so stale UI state can
      no longer resurrect purged artifacts.
    - Deletes the canonical per-stage artifacts listed in ``RESET_PIPELINE_ARTIFACTS``.
    - When ``hard=True``, also removes the entire ``dashboard_rust_enriched``,
      ``checkpoint_b`` and ``checkpoint_c`` staging directories.

    Returns the number of files removed. Safe to call repeatedly; missing files
    are skipped silently.
    """
    removed = 0
    for key in ('vector_upload', 'image_upload', 'last_vector_fmt', 'last_ingest_mode', '_geometry_stage', 'stage_c_export_path', '_last_proc_exit', '_last_proc_log', '_last_proc_section', 'execution_log'):
        st.session_state.pop(key, None)
    st.session_state['execution_log'] = 'Pipeline reset. Ready for new ingest.'
    st.session_state['is_processing'] = False
    cancel_active_subprocess()
    for artifact in RESET_PIPELINE_ARTIFACTS:
        try:
            if artifact.is_file():
                artifact.unlink()
                removed += 1
        except OSError as exc:
            logger.warning('reset_pipeline: could not remove %s (%s)', artifact, exc)
    if hard:
        for sub in (RUST_OUT_DIR, CHECKPOINT_B_DIR, CHECKPOINT_C_DIR):
            try:
                if sub.is_dir():
                    shutil.rmtree(sub)
                    removed += 1
            except OSError as exc:
                logger.warning('reset_pipeline: could not remove dir %s (%s)', sub, exc)
        for sub in (RUST_OUT_DIR, CHECKPOINT_B_DIR, CHECKPOINT_C_DIR):
            sub.mkdir(parents=True, exist_ok=True)
    return removed

def _mtime(path: Path) -> float:
    """File mtime or 0.0 when absent (used for staleness comparisons)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0

def _fresh_against_input(*artifacts: Path) -> bool:
    """True only if every artifact exists AND is at least as new as INPUT_JSON.

    Why: a re-ingest bumps INPUT_JSON's mtime; downstream gates built on older
    artifacts must re-lock so navigation never serves stale prior-run outputs.
    """
    input_mtime = _mtime(INPUT_JSON)
    return all((a.is_file() and _mtime(a) >= input_mtime for a in artifacts))

def gate_unlocked(step: int) -> bool:
    """Geometry-only gate: unlock when INPUT_JSON exists (staleness-aware)."""
    if step == STEP_GEOMETRY:
        return INPUT_JSON.is_file()
    return False


def require_gate(step: int, label: str) -> None:
    """Hard-stop page render when prerequisite artifacts are absent."""
    if gate_unlocked(step):
        return
    st.warning(f'**{label} locked** — complete the prior step or use the Shortcut Ingress upload on this page.')
    st.stop()

def render_navigation_guard() -> None:
    """Sidebar warning while a cancellable subprocess is active."""
    if st.session_state.get('is_processing'):
        st.sidebar.markdown('<div class="nav-lock">🔒 Pipeline active — finish or cancel before switching stages.</div>', unsafe_allow_html=True)

def pipeline_active_step() -> int:
    """1=ingress, 2=geometry run, 3=export ready."""
    if ds_fresh_export():
        return 3
    if resolve_enriched_json(INPUT_JSON) is not None:
        return 2
    if INPUT_JSON.is_file():
        return 2
    return 1


def ds_fresh_export() -> bool:
    return STAGE_V2_JSON.is_file() and _fresh_against_input(STAGE_V2_JSON)


def render_sidebar_shell() -> None:
    """Shared sidebar: pipeline status, advanced settings, execution logs."""
    render_navigation_guard()
    st.sidebar.markdown("### Nexus-Forge Cyto")
    st.sidebar.caption("خط لوله هندسی آفلاین")
    step = pipeline_active_step()
    for num, label in ((1, "دریافت"), (2, "هندسه"), (3, "خروجی")):
        mark = "✅" if num < step else ("🔵" if num == step else "⬜")
        st.sidebar.markdown(f"{mark} {label}")
    if INPUT_JSON.is_file():
        fmt = st.session_state.get("last_vector_fmt", "—")
        st.sidebar.caption(f"ورودی: {INPUT_JSON.name} · {fmt}")
    st.sidebar.divider()
    with st.sidebar.expander('Advanced Settings', expanded=False):
        st.session_state['fill_opacity'] = st.slider(
            'Polygon opacity', 0.1, 0.9, float(st.session_state['fill_opacity']), 0.05
        )
    with st.sidebar.expander('Execution Logs', expanded=False):
        st.code(st.session_state.get('execution_log', 'No runs yet.'), language='text')
    st.sidebar.divider()
    with st.sidebar.expander('Pipeline Controls', expanded=False):
        st.caption('Remove all stage artifacts and return to a clean geometry ingest state.')
        col_soft, col_hard = st.columns(2)
        if col_soft.button('Reset', help='Delete stage artifacts but keep staging directories.'):
            n = reset_pipeline(hard=False)
            st.toast(f'Reset complete — {n} artifact(s) removed.', icon='🧹')
            st.rerun()
        if col_hard.button('Hard Reset', help='Also wipe dashboard_rust_enriched / checkpoint_b / checkpoint_c.'):
            n = reset_pipeline(hard=True)
            st.toast(f'Hard reset complete — {n} item(s) removed.', icon='🧹')
            st.rerun()

def truncate_log_line(line: str) -> str:
    encoded = line.encode('utf-8', errors='replace')
    if len(encoded) <= MAX_LOG_LINE_BYTES:
        return line
    suffix = LOG_TRUNCATION_SUFFIX.encode('utf-8')
    budget = max(0, MAX_LOG_LINE_BYTES - len(suffix))
    return encoded[:budget].decode('utf-8', errors='ignore') + LOG_TRUNCATION_SUFFIX

def append_execution_log(section: str, log: str) -> None:
    prior = st.session_state.get('execution_log', '')
    block = f"=== {section} ===\n{log or '(no output)'}"
    combined = f'{prior}\n\n{block}'.strip() if prior else block
    lines = combined.splitlines()
    if len(lines) > MAX_EXEC_LOG_LINES:
        lines = lines[-MAX_EXEC_LOG_LINES:]
    st.session_state['execution_log'] = '\n'.join(lines)

def purge_partial_artifacts(paths: list[str | Path]) -> None:
    """Remove incomplete stage artifacts after cancellation."""
    for raw in paths:
        p = Path(raw)
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except OSError as exc:
            logger.warning('Partial purge skipped %s: %s', p, exc)

def _assert_no_image_argv(cmd: list[str]) -> None:
    for token in cmd:
        if str(token).lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            raise ValueError('Images never enter subprocess layers.')

def _read_proc_stdout(proc: subprocess.Popen[str], buf: list[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(truncate_log_line(line.rstrip('\n')))
        if len(buf) > MAX_EXEC_LOG_LINES:
            buf.pop(0)

def purge_processing_state(*, force_rerun: bool=False) -> None:
    """Hard-reset cancellable subprocess UI/session keys after terminal or failed runs."""
    st.session_state['is_processing'] = False
    st.session_state['active_proc'] = None
    st.session_state['active_proc_log'] = []
    st.session_state['active_proc_partials'] = []
    st.session_state.pop('active_proc_section', None)
    if force_rerun:
        st.rerun()

def launch_cancellable_subprocess(cmd: list[str], *, env: dict[str, str] | None, section: str, partial_paths: list[Path] | None=None) -> None:
    """Start a tracked subprocess; poll/cancel across Streamlit reruns."""
    try:
        _assert_no_image_argv(cmd)
        env = env or dict(os.environ)
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        log_buf: list[str] = []
        st.session_state['active_proc'] = proc
        st.session_state['active_proc_log'] = log_buf
        st.session_state['active_proc_section'] = section
        st.session_state['active_proc_partials'] = [str(p) for p in partial_paths or []]
        st.session_state['is_processing'] = True
        threading.Thread(target=_read_proc_stdout, args=(proc, log_buf), daemon=True).start()
    except Exception as exc:
        append_execution_log(section, f'Launch failed: {exc}')
        st.session_state['_last_proc_exit'] = -1
        st.session_state['_last_proc_log'] = str(exc)
        st.session_state['_last_proc_section'] = section
        purge_processing_state(force_rerun=True)
    finally:
        if st.session_state.get('active_proc') is None and st.session_state.get('is_processing'):
            st.session_state['is_processing'] = False

def cancel_active_subprocess() -> None:
    """Terminate active PID, purge partial artifacts, reset processing flag."""
    proc: subprocess.Popen[str] | None = st.session_state.get('active_proc')
    try:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        purge_partial_artifacts(st.session_state.get('active_proc_partials', []))
        append_execution_log('Cancelled', 'Operation terminated by user.')
    finally:
        purge_processing_state(force_rerun=True)

def _finalize_subprocess_outcome(code: int, log: str) -> None:
    """Record exit metadata and dissolve processing chrome before error/success UI."""
    section = st.session_state.get('active_proc_section', 'subprocess')
    append_execution_log(section, log)
    st.session_state['_last_proc_exit'] = code
    st.session_state['_last_proc_log'] = log
    st.session_state['_last_proc_section'] = section
    purge_processing_state(force_rerun=True)

def render_cancellation_panel() -> bool:
    """Live status + cancel button. Returns True only while subprocess is actively running."""
    if not st.session_state.get('is_processing'):
        return False
    try:
        proc: subprocess.Popen[str] | None = st.session_state.get('active_proc')
        if proc is None:
            purge_processing_state(force_rerun=True)
            return False
        code = proc.poll()
        if code is not None:
            log = '\n'.join(st.session_state.get('active_proc_log', [])).strip()
            _finalize_subprocess_outcome(code, log)
            return False
        st.info('⏳ Processing Pipeline Stage... Please wait.')
        if st.button('🛑 Cancel Operation', type='primary', key='global_cancel_op'):
            cancel_active_subprocess()
            return False
        st.caption(f"Active: {st.session_state.get('active_proc_section', 'subprocess')}")
        import time
        poll_ms = int(os.environ.get('NEXUS_POLL_MS', '500'))
        time.sleep(max(poll_ms, 100) / 1000.0)
        st.rerun()
        return True
    except Exception as exc:
        append_execution_log('Pipeline Error', str(exc))
        st.session_state['_last_proc_exit'] = -1
        st.session_state['_last_proc_log'] = str(exc)
        st.session_state['_last_proc_section'] = st.session_state.get('active_proc_section', '')
        purge_processing_state(force_rerun=True)
        return False
    finally:
        if st.session_state.get('active_proc') is None:
            st.session_state['is_processing'] = False

def consume_last_subprocess_result(expected_section: str | None=None) -> tuple[int | None, str]:
    """Pop the last subprocess outcome, but only if it belongs to this stage.

    Why: pages share one result slot; switching pages mid-run could let stage B
    consume stage A's exit code and render a false banner. Non-matching results
    are left intact for their owning page to consume.
    """
    section = st.session_state.get('_last_proc_section')
    if expected_section is not None and section is not None and (not str(section).startswith(expected_section)):
        return (None, '')
    code = st.session_state.pop('_last_proc_exit', None)
    log = st.session_state.pop('_last_proc_log', '')
    st.session_state.pop('_last_proc_section', None)
    return (code, log)

def resolve_executable(name: str) -> str | None:
    found = shutil.which(name)
    return os.path.abspath(found) if found else None

def resolve_enriched_json(input_path: Path) -> Path | None:
    candidate = RUST_OUT_DIR / f'{input_path.stem}_enriched.json'
    return candidate if candidate.is_file() else None

def normalize_vector_upload_to_json_bytes(filename: str, raw: bytes) -> tuple[bytes, str]:
    """Normalize vector uploads via the same gold-segment contract as the HTTP API."""
    gold = gold_segment_from_vector_bytes(filename, raw)
    adapter = gold.get("annotation_adapter") or {}
    fmt = str(adapter.get("format") or detect_vector_format(filename, raw))
    return (json.dumps(gold, ensure_ascii=False).encode("utf-8"), fmt)

def write_vector_bootstrap(vector_upload) -> None:
    """Expert vector → INPUT_JSON with M-01 purge."""
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    normalized_json, src_fmt = normalize_vector_upload_to_json_bytes(vector_upload.name, vector_upload.getvalue())
    INPUT_JSON.write_bytes(normalized_json)
    st.session_state['last_vector_fmt'] = src_fmt
    st.session_state['vector_upload'] = None
    st.session_state['image_upload'] = None
    st.session_state['last_ingest_mode'] = 'vector'

def write_input_json_shortcut(upload) -> None:
    """Shortcut gate: accept pre-built bootstrap JSON."""
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_JSON.write_bytes(upload.getvalue())
    st.session_state['last_vector_fmt'] = FORMAT_JSON_STAGE
    st.session_state['vector_upload'] = None
    st.session_state['image_upload'] = None


def write_image_bootstrap(image_upload) -> None:
    """Patch image → SAM 3 subprocess → INPUT_JSON gold segment."""
    import subprocess
    import tempfile

    name = image_upload.name or "patch.png"
    suffix = Path(name).suffix or ".png"

    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=INGEST_DIR) as tmp:
        tmp.write(image_upload.getvalue())
        tmp_path = Path(tmp.name)

    gold_tmp = INGEST_DIR / f"_sam3_{tmp_path.stem}.json"
    script = REPO_ROOT / "scripts" / "image_to_gold.py"
    proc = subprocess.run(
        [python_executable(), str(script), str(tmp_path), str(gold_tmp)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "SAM 3 subprocess failed").strip()
        if proc.returncode == 5:
            raise ValueError("هیچ هسته‌ای در تصویر شناسایی نشد.")
        raise ValueError(detail[:2000])

    if not gold_tmp.is_file():
        raise ValueError("SAM 3 completed but gold segment was not written.")

    gold = json.loads(gold_tmp.read_text(encoding="utf-8"))
    INPUT_JSON.write_bytes(gold_tmp.read_bytes())
    try:
        gold_tmp.unlink(missing_ok=True)
    except OSError:
        pass

    adapter = gold.get("annotation_adapter") or {}
    st.session_state["last_vector_fmt"] = str(adapter.get("format") or FORMAT_SAM3_NUCLEI)
    st.session_state["vector_upload"] = None
    st.session_state["image_upload"] = None
    st.session_state["last_ingest_mode"] = "image"

def rust_binary_built() -> bool:
    """Safety check: prebuilt release binary present so we skip per-run `cargo` recompile."""
    return rust_release_bin().is_file()

def build_rust_geometry_cmd(input_path: Path) -> list[str]:
    """Invoke the compiled Rust enrichment core (native on Windows and Linux)."""
    args = ['--input-dir', str(input_path.parent), '--output-dir', str(RUST_OUT_DIR)]
    if rust_binary_built():
        return [str(rust_release_bin()), *args]
    cargo_exec = resolve_executable('cargo') or 'cargo'
    return [cargo_exec, 'run', '--release', '--bin', 'nexus-forge-cyto-batch', '--', *args]

def build_graph_export_cmd(enriched_json: Path) -> tuple[list[str], dict[str, str]]:
    """Delegate headless schema export to the shared pipeline_core contract."""
    return build_headless_export_cmd(enriched_json, output_dir=OUTPUT_DIR, schema_path_override=SCHEMA_PATH)

def ensure_qupath_export() -> Path | None:
    """Build QuPath GeoJSON from stage_v2 or Rust enriched when available."""
    source = STAGE_V2_JSON if STAGE_V2_JSON.is_file() else resolve_enriched_json(INPUT_JSON)
    if source is None or not source.is_file():
        return None
    try:
        return write_qupath_from_enriched(source, QUPATH_GEOJSON)
    except Exception as exc:
        logger.warning("QuPath export skipped: %s", exc)
        return None


def render_export_console(key_prefix: str='console') -> None:
    """Modular export buttons with lazy, on-demand byte loading.

    Why lazy: eagerly reading every artifact (incl. large graph/stage files) on each
    rerun caused M-01 memory pressure and risked serving stale bytes. Bytes are only
    read when the user presses 'Prepare', and the prepared download is invalidated when
    the file's mtime changes.
    """
    enriched = resolve_enriched_json(INPUT_JSON)
    ensure_qupath_export()
    artifacts = [
        ('INPUT_JSON', INPUT_JSON, 'application/json'),
        ('Rust Enriched JSON', enriched, 'application/json'),
        ('stage_v2_final.json', STAGE_V2_JSON, 'application/json'),
        ('QuPath GeoJSON', QUPATH_GEOJSON, 'application/geo+json'),
    ]
    for label, path, mime in artifacts:
        slot = f'{key_prefix}_{label}'
        if not path or not path.is_file():
            st.button(f'{label} — pending', disabled=True, key=f'{slot}_pending')
            continue
        state_key = f'_dl_ready_{slot}'
        current_mtime = _mtime(path)
        ready = st.session_state.get(state_key)
        if ready is not None and ready != current_mtime:
            st.session_state.pop(state_key, None)
            ready = None
        if ready is None:
            if st.button(f'Prepare {label}', key=f'{slot}_prep'):
                st.session_state[state_key] = current_mtime
                st.rerun()
        else:
            data = read_file_bytes_if_exists(path)
            if data:
                st.download_button(label, data=data, file_name=path.name, mime=mime, key=slot)
            else:
                st.session_state.pop(state_key, None)
                st.button(f'{label} — pending', disabled=True, key=f'{slot}_pending')
