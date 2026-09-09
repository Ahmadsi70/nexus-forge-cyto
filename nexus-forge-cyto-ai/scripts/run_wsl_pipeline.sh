#!/usr/bin/env bash
# Run **from repo root on Windows**: `wsl bash scripts/run_wsl_pipeline.sh`
# Assumes **Ubuntu/WSL** with Rust (`cargo`) and **Python 3** available inside Linux.
# Uses **`libmath_core.so`** under **`${ROOT}/mojo`** via **`CYTO_MOJO_LIB_DIR`** + **`LD_LIBRARY_PATH`**.
#
# If you cloned on `/mnt/c` and see **`$'\r'`** errors: `sed -i 's/\r$//' scripts/run_wsl_pipeline.sh`
set -euo pipefail

export PATH="${HOME}/.cargo/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [[ -z "${MONUSEG_XML:-}" && -z "${CYTO_MONUSEG_XML:-}" ]]; then
  echo "FATAL: set MONUSEG_XML or CYTO_MONUSEG_XML to a MoNuSeg annotation XML path." >&2
  exit 2
fi

XML_INPUT="${MONUSEG_XML:-${CYTO_MONUSEG_XML}}"
export CYTO_MONUSEG_XML="${XML_INPUT}"
export CYTO_MOJO_LIB_DIR="${CYTO_MOJO_LIB_DIR:-${ROOT}/mojo}"
export LD_LIBRARY_PATH="${CYTO_MOJO_LIB_DIR}:${LD_LIBRARY_PATH:-}"

for _libm in /lib/x86_64-linux-gnu/libm.so.6 /usr/lib/x86_64-linux-gnu/libm.so.6; do
  if [[ -f "${_libm}" ]]; then
    export LD_PRELOAD="${_libm}${LD_PRELOAD:+:${LD_PRELOAD}}"
    break
  fi
done

export CYTO_SKIP_TECH_REPORT="${CYTO_SKIP_TECH_REPORT:-1}"

echo "ROOT=${ROOT}"
echo "CYTO_MOJO_LIB_DIR=${CYTO_MOJO_LIB_DIR}"
echo "LD_PRELOAD=${LD_PRELOAD:-}"
echo "CYTO_SKIP_TECH_REPORT=${CYTO_SKIP_TECH_REPORT}"
echo "XML_INPUT=${XML_INPUT}"

mkdir -p "${ROOT}/tests/validation"
python3 <<PY
from pathlib import Path
import sys

ROOT = Path("${ROOT}")
XML = Path("${XML_INPUT}")
GOLD = ROOT / "tests" / "validation" / "gold_segment.json"
sys.path.insert(0, str(ROOT / "services"))
from nexus_core.coord_adapter import write_gold_segment_json

if not XML.is_file():
    sys.stderr.write(f"FATAL: XML not found: {XML}\n")
    sys.exit(2)
write_gold_segment_json(XML, GOLD)
print(f"OK: wrote {GOLD}")
PY

export CYTO_GOLD_SEGMENT_JSON="${ROOT}/tests/validation/gold_segment.json"

echo ""
echo "=== [1/2] cargo test --test test_final_flow ==="
cargo test --test test_final_flow -- --nocapture

echo ""
echo "=== [2/2] tests/integration/debug_monuseg_run.py ==="
python3 "${ROOT}/tests/integration/debug_monuseg_run.py" --input "${XML_INPUT}"
