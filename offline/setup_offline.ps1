# Setup script — fully offline installation (no network required).
# Run from the project root:  powershell -ExecutionPolicy Bypass -File offline\setup_offline.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "=== 1. Installing Python deps from local wheels (offline) ==="
python -m pip install --no-index --find-links "$Root\offline\wheels" -r "$Root\nexus-forge-cyto-core\python\requirements-fusion.txt"

Write-Host "=== 2. Building the Rust geometry engine (offline, vendored crates) ==="
Push-Location $Root
cargo build -p nexus-forge-cyto-core --release --bin nexus-core-cli
Pop-Location

Write-Host ""
Write-Host "Done. Now run the demo:"
Write-Host "  python nexus-forge-cyto-core\python\run_fusion_demo.py <image_path>"
