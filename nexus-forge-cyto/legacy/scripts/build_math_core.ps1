# Builds Phase-2 shared library for Rust FFI (**why**: Rust expects CYTO_MOJO_LIB_DIR to contain math_core DLL).
# Prerequisites: Modular MAX installed with `mojo` on PATH.
# Official CLI reference: https://docs.modular.com/mojo/cli/build
#
# MSVC note: test/binary crates link `math_core.lib` (import library). Place **`math_core.lib`** next to the DLL when the Mojo
# toolchain emits it.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$MojoFile = Join-Path $RepoRoot "mojo\math_core.mojo"
$OutDir = Join-Path $RepoRoot "mojo_dynlib"
$DllOut = Join-Path $OutDir "math_core.dll"

if (-not (Test-Path $MojoFile)) {
    Write-Error "Missing Mojo source: $MojoFile"
}

$mojo = Get-Command mojo -ErrorAction SilentlyContinue
if (-not $mojo) {
    Write-Error @"
'mojo' was not found on PATH. Install Modular MAX and ensure the CLI is available, then re-run:
  https://docs.modular.com/mojo/cli/build
"@
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Push-Location (Join-Path $RepoRoot "mojo")
try {
    # Shared library for Rust dylib link + runtime load (Windows: math_core.dll).
    & mojo build math_core.mojo --emit shared-lib -o $DllOut
    if (-not (Test-Path $DllOut)) {
        Write-Error "Build reported success but output missing: $DllOut"
    }
    Write-Host "OK: $DllOut"
}
finally {
    Pop-Location
}
