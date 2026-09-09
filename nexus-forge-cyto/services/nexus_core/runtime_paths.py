"""
Portable runtime path contract for Nexus-Forge cytology pipeline stages.

Why: every service resolves artifacts relative to the repository root so exports
remain machine-independent and deploy cleanly across Windows/Linux hosts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Resolve repository root depth-independently (env override → sentinel walk → fallback)."""
    override = os.environ.get("NEXUS_REPO_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "services").is_dir() and (parent / "schemas").is_dir():
            return parent
    # Fallback: <repo>/services/nexus_core/runtime_paths.py
    return here.parents[2]


def workspace_root() -> Path:
    """Return workspace root (parent repo) for shared .env discovery."""
    return repo_root().parent


def production_output_dir(*, create: bool = True) -> Path:
    """Canonical runtime directory: <repo>/tmp/production_output/."""
    override = os.environ.get("NEXUS_OUTPUT_DIR", "").strip()
    out = Path(override) if override else repo_root() / "tmp" / "production_output"
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def default_input_json() -> Path:
    """Resolve Stage-A input JSON from env or existing checkpoint handoff files."""
    override = os.environ.get("NEXUS_INPUT_JSON", "").strip()
    if override:
        return Path(override)
    out = production_output_dir()
    ingest = out / "dashboard_ingest" / "uploaded_tissue.json"
    checkpoint = out / "stage_v2_final.json"
    if ingest.is_file():
        return ingest
    if checkpoint.is_file():
        return checkpoint
    ingest.parent.mkdir(parents=True, exist_ok=True)
    return ingest


def schema_path() -> Path:
    """JSON schema path for enriched graph v2 validation."""
    override = os.environ.get("NEXUS_SCHEMA_PATH", "").strip()
    return Path(override) if override else repo_root() / "schemas" / "enriched_graph_v2.schema.json"


def hovernet_instances_path() -> Path:
    """Optional HoverNet projection instances payload."""
    override = os.environ.get("NEXUS_HOVERNET_INSTANCES", "").strip()
    return (
        Path(override)
        if override
        else production_output_dir() / "hovernet_instances.json"
    )


def python_executable() -> str:
    """Interpreter for SAM 3 subprocesses; honors NEXUS_PYTHON when set on GPU hosts."""
    override = os.environ.get("NEXUS_PYTHON", "").strip()
    if override and Path(override).is_file():
        return override
    return sys.executable


def is_windows_host() -> bool:
    """True when the orchestrator runs on a native Windows host."""
    return os.name == "nt"


def portable_artifact_ref(path: Path) -> str:
    """Serialize artifact location as repo-relative POSIX path when possible."""
    resolved = path.resolve()
    root = repo_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name
