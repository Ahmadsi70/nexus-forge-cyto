"""
Rust FFI bridge — replaces subprocess-based enrichment with native ctypes calls.

Why: Subprocess per request incurs ~100ms+ overhead (process spawn, filesystem I/O),
while direct FFI calls are ~1µs and avoid temp file leaks.

Supports three modes:
1. **shared_library** — Loads `nexus_forge_cyto_ai.dll`/`.so` via ctypes (fastest).
2. **subprocess** — Falls back to spawning the Rust binary (original behavior).
3. **auto** — Tries shared_library first, falls back to subprocess.

Set `NEXUS_RUST_BRIDGE=shared_library` to force FFI mode (requires compiled Rust cdylib).
Set `NEXUS_RUST_BRIDGE=subprocess` to force legacy subprocess mode.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from nexus_core.runtime_paths import is_windows_host, repo_root


def _bridge_mode() -> str:
    """Returns the resolved bridge mode: 'shared_library' or 'subprocess'."""
    mode = os.environ.get("NEXUS_RUST_BRIDGE", "auto").strip().lower()

    if mode == "shared_library":
        return "shared_library"
    elif mode == "subprocess":
        return "subprocess"

    # auto: detect shared library
    lib_name = _find_shared_lib()
    if lib_name is not None:
        return "shared_library"
    return "subprocess"


def _find_shared_lib() -> str | None:
    """Locate the compiled Rust cdylib in target/release/ or target/debug/."""
    base = repo_root()
    candidates = []
    if is_windows_host():
        candidates.extend([
            base / "target" / "release" / "nexus_forge_cyto_ai.dll",
            base / "target" / "debug" / "nexus_forge_cyto_ai.dll",
        ])
    else:
        candidates.extend([
            base / "target" / "release" / "libnexus_forge_cyto_ai.so",
            base / "target" / "debug" / "libnexus_forge_cyto_ai.so",
        ])

    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _ffi_enrich(gold_segment: dict[str, Any]) -> dict[str, Any]:
    """Enrich gold segment via ctypes FFI (fast path)."""
    # Lazy import — ctypes is stdlib
    import ctypes

    lib_path = _find_shared_lib()
    if lib_path is None:
        raise RuntimeError("Rust shared library not found — build with 'cargo build --release' first.")

    lib = ctypes.CDLL(lib_path)

    # Define C function signature
    # write_enriched_geometric_output(raw_bytes: *const u8, len: usize, output_path: *const c_char) -> int
    try:
        func = lib.process_enriched_json
        func.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        func.restype = ctypes.c_int
    except AttributeError:
        # Function not exported, fallback to subprocess
        raise RuntimeError("FFI function not found in shared library — rebuild with --features ffi-bridge")

    # Serialize input
    raw_bytes = json.dumps(gold_segment).encode("utf-8")

    # Create temp output path
    work = repo_root() / "tmp" / "ffi_jobs" / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=True)
    output_path = work / "enriched.json"

    try:
        rc = func(
            raw_bytes,
            len(raw_bytes),
            str(output_path).encode("utf-8"),
            len(str(output_path)),
        )
        if rc != 0:
            raise RuntimeError(f"Rust FFI enrichment failed with code {rc}")

        if not output_path.is_file():
            raise RuntimeError("Rust FFI enrichment produced no output file")

        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        # Cleanup temp directory
        shutil.rmtree(work, ignore_errors=True)


def _subprocess_enrich(gold_segment: dict[str, Any]) -> dict[str, Any]:
    """Enrich gold segment via subprocess (legacy path)."""
    work = repo_root() / "tmp" / "production_output" / "api_jobs" / uuid.uuid4().hex
    ingest = work / "ingest"
    out_dir = work / "rust_out"
    ingest.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        gold_path = ingest / "gold_segment.json"
        gold_path.write_text(json.dumps(gold_segment, ensure_ascii=False) + "\n", encoding="utf-8")

        bin_path = _rust_binary_path()
        if bin_path.is_file():
            cmd = [str(bin_path), "--input-dir", str(ingest), "--output-dir", str(out_dir)]
        else:
            cargo = shutil.which("cargo")
            if cargo is None:
                raise RuntimeError(
                    "Rust binary not found and 'cargo' is not in PATH. "
                    "Run 'cargo build --release --bin nexus-forge-cyto-batch' first."
                )
            cmd = [
                cargo,
                "run",
                "--release",
                "--bin",
                "nexus-forge-cyto-batch",
                "--",
                "--input-dir",
                str(ingest),
                "--output-dir",
                str(out_dir),
            ]

        env = os.environ.copy()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root()),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,  # 2-minute timeout to prevent runaway processes
        )
        enriched_path = out_dir / "gold_segment_enriched.json"
        if proc.returncode != 0 or not enriched_path.is_file():
            detail = (proc.stderr or proc.stdout or "unknown error").strip()
            raise RuntimeError(f"Rust enrichment failed: {detail[:2000]}")
        return json.loads(enriched_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _rust_binary_path() -> Path:
    """Resolved release batch binary (.exe on Windows)."""
    base = repo_root() / "target" / "release" / "nexus-forge-cyto-batch"
    if is_windows_host():
        exe = base.with_suffix(".exe")
        return exe if exe.is_file() else base
    return base


def run_rust_enrichment(gold_segment: dict[str, Any]) -> dict[str, Any]:
    """Enrich a gold segment using Rust (geometry + κ engine).

    Automatically selects the fastest available bridge mode.
    """
    mode = _bridge_mode()

    if mode == "shared_library":
        try:
            return _ffi_enrich(gold_segment)
        except (RuntimeError, ImportError, OSError) as exc:
            # FFI failed, fall back to subprocess with warning
            print(f"[WARN] FFI enrichment failed ({exc}), falling back to subprocess.")
            return _subprocess_enrich(gold_segment)

    return _subprocess_enrich(gold_segment)