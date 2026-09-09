//! Phase-3 integration: mmap session (Rust) ⇄ Mojo `math_core` κ pipeline.
//!
//! **Why RAII session guard**: `cyto_neighbor_session_release` MUST run exactly once—even when the
//! Mojo kernel rejects inputs—preventing orphaned `mmap` slabs on edge SRAM budgets.

use std::ffi::{CStr, CString};
use std::ptr;

use crate::core::{CytoNeighborMatrixView, CytoStatus, CYTO_MAX_FEATURE_DIM_D};
use crate::cyto_neighbor_matrix_view;
use crate::cyto_neighbor_session_open;
use crate::cyto_neighbor_session_release;
use crate::mojo_bind::{invoke_mojo_curvature, mojo_work_f32_elems};
use crate::CytoNeighborSession;

/// Owned mmap session handle; frees on drop (why: integration cannot rely on callers for release).
///
/// SAFETY invariant: constructed only via [`Self::try_open_legacy`] from a successful
/// `cyto_neighbor_session_open`; never duplicated without duplicating mmap ownership semantics.
pub struct CytoNeighborSessionGuard {
    raw: *mut CytoNeighborSession,
}

impl CytoNeighborSessionGuard {
    /// Opens Phase-1 mmap session backing the `(32, D)` dense matrix.
    ///
    /// **Why CString here only**: Path conversion is pre-Mojo and outside the hot zero-copy path.
    pub fn try_open_legacy(path: &CStr, dims_d: u32) -> Result<Self, CytoStatus> {
        let mut raw: *mut CytoNeighborSession = ptr::null_mut();
        let code = unsafe { cyto_neighbor_session_open(path.as_ptr(), dims_d, &mut raw) };
        let status = CytoStatus::from_u32_transport(code);
        if status != CytoStatus::CytoSuccess {
            return Err(status);
        }
        if raw.is_null() {
            return Err(CytoStatus::CytoErrNullPtr);
        }
        Ok(Self { raw })
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut CytoNeighborSession {
        self.raw
    }
}

impl Drop for CytoNeighborSessionGuard {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cyto_neighbor_session_release(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// Deterministic entry point with explicit `D` (matches Phase-1 `open` contract).
///
/// **Zero allocation during Mojo**: scratch is `reserve_exact`/`resize` *before* `invoke_mojo_curvature`.
pub fn analyze_slide_with_dims(
    wsi_path: &str,
    dims_d: u32,
    optical_scale: f32,
) -> Result<(f32, CytoStatus), CytoStatus> {
    if !(1..=CYTO_MAX_FEATURE_DIM_D).contains(&dims_d) {
        return Err(CytoStatus::CytoErrInvalidLen);
    }

    let c_path = CString::new(wsi_path).map_err(|_| CytoStatus::CytoErrIo)?;
    let guard = CytoNeighborSessionGuard::try_open_legacy(c_path.as_c_str(), dims_d)?;

    let mut view = CytoNeighborMatrixView::empty();
    let vcode =
        unsafe { cyto_neighbor_matrix_view(guard.as_ptr().cast_const(), ptr::addr_of_mut!(view)) };
    let vstat = CytoStatus::from_u32_transport(vcode);
    if vstat != CytoStatus::CytoSuccess {
        return Err(vstat);
    }

    let scratch_len = mojo_work_f32_elems(dims_d).ok_or(CytoStatus::CytoErrInvalidLen)?;
    let mut scratch: Vec<f32> = Vec::new();
    scratch
        .try_reserve_exact(scratch_len)
        .map_err(|_| CytoStatus::CytoErrOom)?;
    scratch.resize(scratch_len, 0.0);

    let mut kappa: f32 = 0.0;
    let mut clinical_raw: u32 = 0;

    let mcode = unsafe {
        invoke_mojo_curvature(
            ptr::addr_of!(view),
            optical_scale,
            ptr::addr_of_mut!(kappa),
            ptr::addr_of_mut!(clinical_raw),
            scratch.as_mut_ptr(),
            scratch.len(),
        )
    };
    let mstat = CytoStatus::from_u32_transport(mcode);
    if mstat != CytoStatus::CytoSuccess {
        return Err(mstat);
    }

    let clinical = CytoStatus::from_u32_transport(clinical_raw);
    match clinical {
        CytoStatus::CytoClinicalMalignant | CytoStatus::CytoClinicalNormal => Ok((kappa, clinical)),
        _ => Err(CytoStatus::CytoErrMojoUnknownTransport),
    }
}

/// Primary operator entry (Phase-3): reads `EDGE_CYTO_DIMS_D` for the dense column count `D`.
///
/// **Why env indirection**: Phase-1 `cyto_neighbor_session_open` requires `D` before mmap validation;
/// edge fleet firmware sets this once per tensor manifest so the public API stays `(path, scale)`.
pub fn analyze_slide(wsi_path: &str, optical_scale: f32) -> Result<(f32, CytoStatus), CytoStatus> {
    let raw =
        std::env::var("EDGE_CYTO_DIMS_D").map_err(|_| CytoStatus::CytoErrIntegrationConfig)?;
    let dims_d: u32 = raw
        .parse()
        .map_err(|_| CytoStatus::CytoErrIntegrationConfig)?;
    analyze_slide_with_dims(wsi_path, dims_d, optical_scale)
}

#[cfg(test)]
mod pipe_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn integration_runs_rust_kappa_without_mojo_library() {
        let d: u32 = 4;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let elems = (64_usize) * (d as usize);
        for i in 0..elems {
            tmp.write_all(&(i as f32).to_ne_bytes()).unwrap();
        }
        tmp.flush().unwrap();
        let res = analyze_slide_with_dims(tmp.path().to_str().unwrap(), d, 1.0_f32);
        assert!(res.is_ok(), "rust κ engine should run without libmath_core: {res:?}");
    }
}
