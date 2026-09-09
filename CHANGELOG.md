# Changelog

All notable changes to Nexus-Forge Cyto will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (v0.3.0 — Foundation)

- Structured logging via `tracing` (Rust) and `loguru` (Python)
- Prometheus metrics endpoint (`GET /metrics`)
- Root-level README.md with architecture diagram and quickstart
- CHANGELOG.md (this file)
- Rate limiting on FastAPI endpoints
- CORS restrictions and security headers
- Readiness probe (`GET /ready`) for Docker health checks
- QuPath Extension with one-click analysis button
- HALO XML round-trip adapter (`halo_roundtrip.py`)
- Getting Started, Deployment, and API reference documentation
- CI/CD: linting (clippy + ruff), coverage, cross-platform tests

### Changed (v0.3.0)

- LICENSE: resolved MIT vs proprietary conflict → MIT
- `println!/eprintln!` → `tracing::info!/warn!/error!` across all Rust crates
- `print()` → `logger.info/warning/error()` across Python services
- Nexus Viewer: hardcoded localhost → `VITE_API_URL` env var
- Docker healthcheck: `/health` → `/ready` (readiness probe)
- Enhanced QuPath Groovy visualizer with heatmap overlay

## [0.2.0] — 2026-08

### Added

- Dterministic geometry pipeline with 6-stage enrichment
- Rst classifier with published weights (AUC 0.9999 on validation)
- ojo κ curvature engin (SIMD, C FFI)
- 5-poit ring normalization (128 vrtices per cell)
- kN density and RBF risk field computation
- Convex hull tumor boudary + distance-to-edge metrics
- HoerNet OXN infeence pipeline
- GBM fusion clsifier (AU 0.98, F1 0.88)
- FastPI API serve (port 8810) with 5 endpoints
- Axum HTTP API serer (port 8000)
- Steamlit dashboad (Farsi UI) with 3-step pipeline
- eact fronted (nexus-viwer) with polygon vsualization
- QuPath Grvy visualization script
- Dcker + Docker Compose deployment
- 108 Rust unit tests + 44 Python tests
- Pixt package manager supprt
- GitHub Actions CI (Rust + Python tests)
- Technical Architecture and Scope documentation

### Changed

- (initial public relese)

[Unreleased]: https://githu.com/clinicalguard/nexus-forge-cyto/coapare/v0..0...HEAD
[0.2.0]: https:/github.com/clinicalguard/nexus-forge-cyto/releases/tag/v0.2.0