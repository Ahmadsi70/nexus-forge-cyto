//! κ curvature bridge — loads the compiled Mojo dynamic library if available, 
//! falling back to pure Rust `kappa_engine` for development environments.

use crate::core::CytoNeighborMatrixView;
use crate::kappa_engine;
use std::sync::OnceLock;

#[cfg(unix)]
const MOJO_LIB_NAME: &str = "libkappa.so";
#[cfg(windows)]
const MOJO_LIB_NAME: &str = "kappa.dll";

static MOJO_LIB: OnceLock<Option<libloading::Library>> = OnceLock::new();

type MojoInvokeFn = unsafe extern "C" fn(
    view: *const CytoNeighborMatrixView,
    optical_scale: f32,
    out_kappa: *mut f32,
    out_clinical: *mut u32,
    scratch: *mut f32,
    scratch_elems: usize,
) -> u32;

type MojoContourFn = unsafe extern "C" fn(
    mask_ptr: *const u8,
    width: i32,
    height: i32,
    out_ptr: *mut f32,
    out_max_len: i32,
) -> i32;

/// Returns scratch `f32` word count (`32*D + 64 + D² + 2*D`).
#[inline]
pub fn mojo_work_f32_elems(dims_d: u32) -> Option<usize> {
    kappa_engine::kappa_work_f32_elems(dims_d)
}

/// Invokes the Mojo SIMD Contour Extractor. Returns number of points.
#[inline]
pub(crate) unsafe fn invoke_mojo_contours(
    mask_ptr: *const u8,
    width: i32,
    height: i32,
    out_ptr: *mut f32,
    out_max_len: i32,
) -> i32 {
    let lib_opt = MOJO_LIB.get_or_init(|| {
        unsafe { libloading::Library::new(MOJO_LIB_NAME).ok() }
    });

    if let Some(lib) = lib_opt {
        if let Ok(func) = unsafe { lib.get::<MojoContourFn>(b"extract_contours_cabi\0") } {
            return func(mask_ptr, width, height, out_ptr, out_max_len);
        }
    }

    // Fallback: Return 0 points if Mojo library/function is missing.
    0
}

/// Invokes the Phase-2 κ kernel. Prefers the compiled Mojo engine, falls back to Rust.
#[inline]
pub(crate) unsafe fn invoke_mojo_curvature(
    view: *const CytoNeighborMatrixView,
    optical_scale: f32,
    out_kappa: *mut f32,
    out_clinical: *mut u32,
    scratch: *mut f32,
    scratch_elems: usize,
) -> u32 {
    let lib_opt = MOJO_LIB.get_or_init(|| {
        unsafe { libloading::Library::new(MOJO_LIB_NAME).ok() }
    });

    if let Some(lib) = lib_opt {
        if let Ok(func) = unsafe { lib.get::<MojoInvokeFn>(b"invoke_kappa_curvature_mojo\0") } {
            // Found the Mojo function! Execute SIMD path.
            return func(view, optical_scale, out_kappa, out_clinical, scratch, scratch_elems);
        }
    }

    // Fallback to pure Rust engine
    kappa_engine::invoke_kappa_curvature(
        view,
        optical_scale,
        out_kappa,
        out_clinical,
        scratch,
        scratch_elems,
    )
}

