//! κ curvature bridge — pure Rust `kappa_engine` (legacy name retained for call-site stability).

use crate::core::CytoNeighborMatrixView;
use crate::kappa_engine;

/// Returns scratch `f32` word count (`32*D + 64 + D² + 2*D`).
#[inline]
pub fn mojo_work_f32_elems(dims_d: u32) -> Option<usize> {
    kappa_engine::kappa_work_f32_elems(dims_d)
}

/// Invokes the Phase-2 κ kernel via the in-crate Rust engine.
#[inline]
pub(crate) unsafe fn invoke_mojo_curvature(
    view: *const CytoNeighborMatrixView,
    optical_scale: f32,
    out_kappa: *mut f32,
    out_clinical: *mut u32,
    scratch: *mut f32,
    scratch_elems: usize,
) -> u32 {
    kappa_engine::invoke_kappa_curvature(
        view,
        optical_scale,
        out_kappa,
        out_clinical,
        scratch,
        scratch_elems,
    )
}
