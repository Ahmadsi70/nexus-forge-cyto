#!/usr/bin/env bash
# Nexus-Forge Cyto — Linux server entry (RunPod / bare metal).
# Usage: bash scripts/run_server.sh {bootstrap|test|api|dashboard|smoke}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export NEXUS_REPO_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}/services"
export NEXUS_OUTPUT_DIR="${NEXUS_OUTPUT_DIR:-${ROOT}/tmp/production_output}"

ensure_pixi() {
  if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | bash
  fi
  export PATH="${HOME}/.pixi/bin:${PATH}"
}

workspace_root() {
  if [[ -f "${ROOT}/../Cargo.toml" ]] && grep -q 'nexus-forge-cyto-geometry' "${ROOT}/../Cargo.toml" 2>/dev/null; then
    cd "${ROOT}/.." && pwd
  else
    echo "${ROOT}"
  fi
}

build_rust() {
  local ws
  ws="$(workspace_root)"
  cd "${ws}"
  if [[ "${ws}" != "${ROOT}" ]]; then
    cargo build --release -p nexus-forge-cyto --bin nexus-forge-cyto-batch
    mkdir -p "${ROOT}/target/release"
    ln -sf "${ws}/target/release/nexus-forge-cyto-batch" "${ROOT}/target/release/nexus-forge-cyto-batch"
  else
    cargo build --release --bin nexus-forge-cyto-batch
  fi
}

rust_bin_path() {
  local ws bin
  ws="$(workspace_root)"
  bin="${ROOT}/target/release/nexus-forge-cyto-batch"
  if [[ -f "${bin}" ]]; then
    echo "${bin}"
  elif [[ -f "${ws}/target/release/nexus-forge-cyto-batch" ]]; then
    echo "${ws}/target/release/nexus-forge-cyto-batch"
  else
    echo "${bin}"
  fi
}

load_dotenv() {
  if [[ -f "${ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT}/.env"
    set +a
  fi
}

app_python() {
  load_dotenv
  if [[ -n "${NEXUS_PYTHON:-}" && -x "${NEXUS_PYTHON}" ]]; then
    echo "${NEXUS_PYTHON}"
  else
    echo "${ROOT}/.pixi/envs/default/bin/python"
  fi
}

run_app() {
  local py
  py="$(app_python)"
  if [[ ! -x "${py}" ]]; then
    ensure_pixi
    py="$(app_python)"
  fi
  export PYTHONPATH="${ROOT}/services"
  export NEXUS_REPO_ROOT="${ROOT}"
  cd "${ROOT}"
  exec "${py}" "$@"
}

cmd="${1:-help}"

case "${cmd}" in
  sam3-bootstrap)
    VENV="${ROOT}/.venv-sam3"
    python3 -m venv --system-site-packages "${VENV}"
    "${VENV}/bin/pip" install -q --upgrade pip
    "${VENV}/bin/pip" install -q \
      "ultralytics>=8.3.237" opencv-python-headless huggingface_hub \
      fastapi uvicorn python-multipart streamlit python-dotenv \
      lxml jsonschema scipy pyarrow Pillow httpx pandas numpy
    # CLIP for SAM3 text prompts — no-deps to keep system-site-packages torch
    "${VENV}/bin/pip" install -q "git+https://github.com/ultralytics/CLIP.git" --no-deps || true
    # SAM3 uses Ultralytics built-in proposal (no StarDist dependency)
    mkdir -p /root/.cache/nexus-forge
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "FATAL: export HF_TOKEN before sam3-bootstrap" >&2
      exit 2
    fi
    "${VENV}/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

dest = Path("/root/.cache/nexus-forge")
dest.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    local_dir=str(dest),
    token=os.environ["HF_TOKEN"],
)
print("sam3.pt", path, round(Path(path).stat().st_size / 1e6, 1), "MB")
PY
    cat > "${ROOT}/.env" <<EOF
NEXUS_SAM3_MODEL_PATH=/root/.cache/nexus-forge/sam3.pt
NEXUS_SAM3_CONF=0.25
NEXUS_PYTHON=${VENV}/bin/python
NEXUS_REPO_ROOT=${ROOT}
EOF
    echo "sam3-bootstrap OK"
    ;;
  bootstrap)
    ensure_pixi
    cd "${ROOT}"
    pixi install
    build_rust
    mkdir -p "${NEXUS_OUTPUT_DIR}"
    bin="$(rust_bin_path)"
    echo "bootstrap OK: $(test -f "${bin}" && echo rust-bin-ready || echo rust-bin-missing) @ ${bin}"
    ;;
  test)
    ensure_pixi
    pixi run test
    if command -v cargo >/dev/null 2>&1 || pixi run cargo --version >/dev/null 2>&1; then
      pixi run cargo test --release --quiet --lib 2>/dev/null || pixi run cargo test --quiet --lib
    fi
    ;;
  api)
    run_app -m uvicorn api_service.main:app --host 0.0.0.0 --port 8810
    ;;
  dashboard)
    run_app -m streamlit run services/interface_service/dashboard_app.py --server.port 8501 --server.address 0.0.0.0
    ;;
  smoke)
    ensure_pixi
    pixi run test
    nohup pixi run api > /tmp/nexus-api.log 2>&1 &
    api_pid=$!
    sleep 3
    if curl -fsS "http://127.0.0.1:8810/health" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      echo "smoke OK: API /health"
    else
      echo "smoke FAIL: API /health" >&2
      kill "${api_pid}" 2>/dev/null || true
      exit 1
    fi
    kill "${api_pid}" 2>/dev/null || true
    ;;
  help|*)
    cat <<EOF
Usage: bash scripts/run_server.sh <command>

  bootstrap   Install pixi deps + release Rust batch binary
  sam3-bootstrap  GPU venv + ultralytics + download sam3.pt (needs HF_TOKEN)
  test        pytest + cargo test (lib)
  api         Start FastAPI on :8810 (uses NEXUS_PYTHON / .venv-sam3 when set)
  dashboard   Start Streamlit on :8501
  smoke       pytest then ephemeral API health check
EOF
    ;;
esac
