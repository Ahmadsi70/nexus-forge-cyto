//! Nexus-Forge Edge Computational Cytology — Step 1 FFI surface.
//!
//! **Why this crate is `cdylib`**: Hosts (Mojo/C++) load one shared object and call `extern "C"` symbols without Rust-specific ABIs.
//! Session setup uses the global allocator exactly once per open/close pair; view queries are strict non-allocating paths.

pub mod classifier;
pub mod core;
pub mod geometric_orchestrator;
mod kappa_engine;
mod mmap_io;
mod mojo_bind;
pub mod pipeline;
pub mod pipeline_parallel;
pub mod spatial;
pub mod vision;
pub mod vision_inference;

pub use crate::classifier::{evaluate_malignancy, malignancy_score, median_f64, CellFeatures};
pub use crate::spatial::{
    calculate_convex_hull, calculate_knn_density, calculate_rbf_field, default_rbf_sigma,
    distance_to_polygon, polygon_area_shoelace_abs, polygon_centroid_xy, polygon_circularity,
    polygon_eccentricity, polygon_perimeter, spatial_nodes_from_rings, ClinicalStatus, SpatialNode,
    DEFAULT_KNN_K,
};
pub use crate::geometric_orchestrator::{
    format_geometric_cell_diagnosis, geometric_results_from_json_bytes,
    write_enriched_geometric_output, GeometricOrchestratorError,
};

pub use crate::core::{
    expected_matrix_byte_len, CytoNeighborMatrixView, CytoStatus, CYTO_MAX_FEATURE_DIM_D,
    CYTO_NEIGHBOR_ROW_COUNT,
};
pub use crate::pipeline::{analyze_slide, analyze_slide_with_dims, CytoNeighborSessionGuard};
pub use crate::pipeline_parallel::{
    parallel_analyze_slide, parallel_analyze_slide_mmap, parallel_analyze_slide_mmap_path,
    warm_parallel_slide_thread_scratch, CytoCellResult,
};

use std::alloc::{self, Layout};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use crate::mmap_io::MappedNeighborMatrixSession;

/// Opaque session handle as seen from C (`CytoNeighborSession*`).
///
/// **Why ZST + pointer pattern**: Prevents C from depending on Rust field layout while still giving a typed pointer for documentation.
#[repr(C)]
pub struct CytoNeighborSession {
    _private: [u8; 0],
}

#[inline]
fn session_layout() -> Layout {
    Layout::new::<MappedNeighborMatrixSession>()
}

/// Maps a read-only WSI/tensor file backing a dense `(32, D)` `f32` neighbor matrix.
///
/// # Safety
///
/// `path` must be a valid, NUL-terminated pointer suitable for `CStr::from_ptr`. `out_session` must be non-null and point to initialized memory only on success.
///
/// **Diagnostics**: On failure the returned `u32` is a [`CytoStatus`] discriminant; decode with
/// [`CytoStatus::from_u32_transport`] and surface [`CytoStatus::operator_hint`] (or `cyto_status_message`) to operators.
///
/// **Why `alloc` instead of `Box`**: Honors the “no `Box` inside FFI entry points” constraint while still running `Drop` for `Mmap` on release.
#[no_mangle]
pub unsafe extern "C" fn cyto_neighbor_session_open(
    path: *const c_char,
    dims_d: u32,
    out_session: *mut *mut CytoNeighborSession,
) -> u32 {
    let code = cyto_neighbor_session_open_impl(path, dims_d, out_session);
    code as u32
}

unsafe fn cyto_neighbor_session_open_impl(
    path: *const c_char,
    dims_d: u32,
    out_session: *mut *mut CytoNeighborSession,
) -> CytoStatus {
    if out_session.is_null() {
        return CytoStatus::CytoErrNullPtr;
    }
    ptr::write(out_session, ptr::null_mut());

    if path.is_null() {
        return CytoStatus::CytoErrNullPtr;
    }

    // SAFETY: caller guarantees `path` points to readable memory; we cap the
    // scan to 4 KiB so an unterminated string can never walk into unmapped pages.
    // We avoid `CStr::from_ptr` directly because that API walks bytes until it
    // finds NUL with no length bound — undefined behaviour on malformed input.
    const MAX_LEN: usize = 4096;
    let bytes = unsafe { std::slice::from_raw_parts(path.cast::<u8>(), MAX_LEN) };
    let cstr = match std::ffi::CStr::from_bytes_until_nul(bytes) {
        Ok(c) => c,
        // No NUL found within 4 KiB — refuse to parse (return Io error).
        Err(_) => return CytoStatus::CytoErrIo,
    };
    let utf8 = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return CytoStatus::CytoErrIo,
    };

    let opened = match MappedNeighborMatrixSession::open(Path::new(utf8), dims_d) {
        Ok(s) => s,
        Err(e) => return e,
    };

    let layout = session_layout();
    let raw = alloc::alloc(layout);
    if raw.is_null() {
        return CytoStatus::CytoErrOom;
    }
    let typed = raw.cast::<MappedNeighborMatrixSession>();
    unsafe { typed.write(opened) };
    ptr::write(out_session, typed.cast::<CytoNeighborSession>());
    CytoStatus::CytoSuccess
}

/// Tears down a session created via [`cyto_neighbor_session_open`].
///
/// # Safety
///
/// `session` must be null or a pointer returned by a successful open call and not freed twice.
#[no_mangle]
pub unsafe extern "C" fn cyto_neighbor_session_release(session: *mut CytoNeighborSession) {
    if session.is_null() {
        return;
    }
    let layout = session_layout();
    let typed = session.cast::<MappedNeighborMatrixSession>();
    unsafe {
        ptr::drop_in_place(typed);
        alloc::dealloc(typed.cast(), layout);
    }
}

/// Fills `out_view` with pointers and strides into the mmap; does not copy matrix bytes.
///
/// # Safety
///
/// `session` must be a live session pointer. `out_view` must be non-null and writable.
#[no_mangle]
pub unsafe extern "C" fn cyto_neighbor_matrix_view(
    session: *const CytoNeighborSession,
    out_view: *mut CytoNeighborMatrixView,
) -> u32 {
    cyto_neighbor_matrix_view_impl(session, out_view) as u32
}

unsafe fn cyto_neighbor_matrix_view_impl(
    session: *const CytoNeighborSession,
    out_view: *mut CytoNeighborMatrixView,
) -> CytoStatus {
    if session.is_null() || out_view.is_null() {
        return CytoStatus::CytoErrNullPtr;
    }
    let inner = unsafe { &*session.cast::<MappedNeighborMatrixSession>() };
    match inner.neighbor_view() {
        Ok(v) => {
            unsafe { out_view.write(v) };
            CytoStatus::CytoSuccess
        }
        Err(e) => e,
    }
}

/// Writes the minimum byte length needed for `(32, dims_d)` in `f32` row-major layout.
///
/// # Safety
///
/// When the function returns success (`0`), non-null `out_bytes` must be a valid writable `u64` address.
///
/// **Why exported**: Lets host code size scratch buffers deterministically without re-implementing shape math.
#[no_mangle]
pub unsafe extern "C" fn cyto_neighbor_expected_payload_bytes(
    dims_d: u32,
    out_bytes: *mut u64,
) -> u32 {
    match expected_matrix_byte_len(dims_d) {
        Some(bytes) if !out_bytes.is_null() => {
            unsafe { out_bytes.write(bytes) };
            CytoStatus::CytoSuccess as u32
        }
        Some(_) => CytoStatus::CytoErrNullPtr as u32,
        None => CytoStatus::CytoErrInvalidLen as u32,
    }
}

/// Returns a static diagnostic string (`NUL`-terminated ASCII) for a [`CytoStatus`] discriminant.
#[no_mangle]
pub extern "C" fn cyto_status_message(status: u32) -> *const c_char {
    const CYTO_SUCCESS: &[u8] = b"CYTO_SUCCESS\0";
    const CYTO_ERR_NULL_PTR: &[u8] = b"CYTO_ERR_NULL_PTR\0";
    const CYTO_ERR_MMAP: &[u8] = b"CYTO_ERR_MMAP\0";
    const CYTO_ERR_INVALID_LEN: &[u8] = b"CYTO_ERR_INVALID_LEN\0";
    const CYTO_ERR_ALIGNMENT: &[u8] = b"CYTO_ERR_ALIGNMENT\0";
    const CYTO_ERR_IO: &[u8] = b"CYTO_ERR_IO\0";
    const CYTO_ERR_OOM: &[u8] = b"CYTO_ERR_OOM\0";
    const CYTO_ERR_INTEGRATION: &[u8] = b"CYTO_ERR_INTEGRATION_CONFIG\0";
    const CYTO_MAL: &[u8] = b"CYTO_CLINICAL_MALIGNANT\0";
    const CYTO_NORM: &[u8] = b"CYTO_CLINICAL_NORMAL\0";
    const CYTO_MOJO_BAD_ARG: &[u8] = b"CYTO_ERR_MOJO_BAD_ARG\0";
    const CYTO_MOJO_BAD_WORK: &[u8] = b"CYTO_ERR_MOJO_BAD_WORK\0";
    const CYTO_MOJO_SHAPE: &[u8] = b"CYTO_ERR_MOJO_SHAPE\0";
    const CYTO_MOJO_LIB_UNSET: &[u8] = b"CYTO_ERR_MOJO_LIBRARY_UNSET\0";
    const CYTO_MOJO_UNK: &[u8] = b"CYTO_ERR_MOJO_UNKNOWN_TRANSPORT\0";
    const CYTO_ERR_UNKNOWN: &[u8] = b"CYTO_ERR_UNKNOWN\0";

    let bytes: &[u8] = match status {
        0 => CYTO_SUCCESS,
        1 => CYTO_ERR_NULL_PTR,
        2 => CYTO_ERR_MMAP,
        3 => CYTO_ERR_INVALID_LEN,
        4 => CYTO_ERR_ALIGNMENT,
        5 => CYTO_ERR_IO,
        6 => CYTO_ERR_OOM,
        8 => CYTO_ERR_INTEGRATION,
        100 => CYTO_MAL,
        101 => CYTO_NORM,
        7010 => CYTO_MOJO_BAD_ARG,
        7020 => CYTO_MOJO_BAD_WORK,
        7030 => CYTO_MOJO_SHAPE,
        7088 => CYTO_MOJO_LIB_UNSET,
        7099 => CYTO_MOJO_UNK,
        _ => CYTO_ERR_UNKNOWN,
    };
    bytes.as_ptr().cast::<c_char>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn ffi_rejects_null_outputs() {
        let st =
            unsafe { cyto_neighbor_session_open_impl(std::ptr::null(), 1, std::ptr::null_mut()) };
        assert_eq!(st, CytoStatus::CytoErrNullPtr);
    }

    #[test]
    fn ffi_roundtrip_view() {
        let d: u32 = 3;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let elems = (CYTO_NEIGHBOR_ROW_COUNT as usize) * (d as usize);
        for i in 0..elems {
            let f = i as f32;
            tmp.write_all(&f.to_ne_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        let path = tmp.path();
        let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

        let mut ses: *mut CytoNeighborSession = std::ptr::null_mut();
        let rc = unsafe {
            cyto_neighbor_session_open_impl(
                c_path.as_ptr(),
                d,
                &mut ses as *mut *mut CytoNeighborSession,
            )
        };
        assert_eq!(rc, CytoStatus::CytoSuccess);

        let mut view = CytoNeighborMatrixView::empty();
        let vrc = unsafe {
            cyto_neighbor_matrix_view_impl(ses, &mut view as *mut CytoNeighborMatrixView)
        };
        assert_eq!(vrc, CytoStatus::CytoSuccess);
        unsafe {
            let sl = crate::core::as_f32_slice(view.data, elems).unwrap();
            assert_eq!(sl[elems - 1], (elems - 1) as f32);
        }

        unsafe { cyto_neighbor_session_release(ses) };
    }

    #[test]
    fn expected_payload_helper() {
        let mut bytes: u64 = 0;
        let st = unsafe { cyto_neighbor_expected_payload_bytes(10, &mut bytes) };
        assert_eq!(st, CytoStatus::CytoSuccess as u32);
        assert_eq!(bytes, 64 * 10 * std::mem::size_of::<f32>() as u64);

        assert_eq!(
            unsafe { cyto_neighbor_expected_payload_bytes(10, std::ptr::null_mut()) },
            CytoStatus::CytoErrNullPtr as u32
        );
    }
}
