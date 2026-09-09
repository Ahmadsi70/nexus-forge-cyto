# Legacy Mojo reference (Phase 1 parity baseline)

The κ (curvature) kernel now lives in **`src/kappa_engine.rs`** (pure Rust).

`reference_math_core.mojo` is kept only as a numeric reference for regression audits.
It is **not** built or linked in Docker, CI, or the dashboard runtime.
