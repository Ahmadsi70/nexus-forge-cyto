#!/usr/bin/env bash
# Run **inside WSL** from repo root: `bash scripts/run_integration_wsl.sh`
# Windows bridge must listen for WSL: `uvicorn main:app --host 0.0.0.0 --port 8810` (+ firewall allow 8810).
# If you cloned on /mnt/c and see `$'\r'` errors, run: `sed -i 's/\r$//' scripts/run_integration_wsl.sh`
set -eu
export PATH="${HOME}/.cargo/bin:${PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
HOST_IP="$(sed -En 's/^nameserver ([0-9.]+)/\1/p' /etc/resolv.conf | head -n1)"
export VISION_BRIDGE_BASE="http://${HOST_IP}:8810"
echo "VISION_BRIDGE_BASE=${VISION_BRIDGE_BASE}"
exec cargo test --test test_final_flow -- --nocapture "$@"
