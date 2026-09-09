//! C-compatible error taxonomy and cache-line-aligned view descriptors for neighbor matrices.

use core::slice;

/// Canonical status codes crossing the FFI boundary (`Result`/`panic` are forbidden in `extern "C"`).
///
/// Unsigned `u32` matches common C enums and avoids sign-extension surprises on 64-bit callers.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CytoStatus {
    CytoSuccess = 0,
    CytoErrNullPtr = 1,
    CytoErrMmap = 2,
    CytoErrInvalidLen = 3,
    CytoErrAlignment = 4,
    CytoErrIo = 5,
    /// Session control block could not be allocated (`alloc`/`Layout` failures).
    CytoErrOom = 6,
    /// Operator misconfiguration (e.g. missing `EDGE_CYTO_DIMS_D` for `analyze_slide`).
    CytoErrIntegrationConfig = 8,
    /// Phase-2 clinical transport codes (Mojo `math_core.mojo`).
    CytoClinicalMalignant = 100,
    CytoClinicalNormal = 101,
    /// Mojo bridge returned `CYTO_FFI_BAD_ARG`.
    CytoErrMojoBadArg = 7010,
    CytoErrMojoBadWork = 7020,
    CytoErrMojoShape = 7030,
    /// Build did not link `math_core` (`CYTO_MOJO_LIB_DIR` unset).
    CytoErrMojoLibraryUnset = 7088,
    /// Unrecognized transport code from the Mojo shared object.
    CytoErrMojoUnknownTransport = 7099,
}

/// Rows in the extracted neighbor tensor; invariant for this pipeline stage.
pub const CYTO_NEIGHBOR_ROW_COUNT: u32 = 64;

/// Aligned descriptor so callers can prefetch/cache-pin metadata without splitting cache lines from payload metadata.
///
/// **Why align(64)**: Keeps concurrent reader metadata on distinct lines versus typical false-sharing with adjacent host structs.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct CytoNeighborMatrixView {
    /// Row-major `[row][col]` with `row_count` rows and `dims_d` columns.
    pub data: *const f32,
    /// Column dimension `D` of the `(32, D)` matrix.
    pub dims_d: u32,
    /// Must equal [`CYTO_NEIGHBOR_ROW_COUNT`].
    pub row_count: u32,
    /// Elements between successive rows (`>= dims_d`; may include tail padding).
    pub stride_elems: u32,
    /// Explicit `u64` alignment pad — MUST mirror Modular Mojo POD (`UInt32 _implicit_pad`).
    pub _implicit_pad_before_byte_length: u32,
    /// Total file-backed length in bytes (for bounds checks by integrators).
    pub byte_length: u64,
    /// WHY: Holds the compiler tail padding explicitly so Rust `sizeof`/`offsets` equal Mojo’s 64-byte view.
    pub _align_reserved_tail: [u8; 32],
}

impl CytoNeighborMatrixView {
    /// Constructs a logically empty view without touching `data` (why: uniform initialization for error paths).
    pub const fn empty() -> Self {
        Self {
            data: core::ptr::null(),
            dims_d: 0,
            row_count: 0,
            stride_elems: 0,
            _implicit_pad_before_byte_length: 0,
            byte_length: 0,
            _align_reserved_tail: [0_u8; 32],
        }
    }
}

impl CytoStatus {
    /// Decodes disciplined `u32` transport codes emitted by FFI layers (Rust `extern "C"` or Mojo).
    #[inline]
    pub fn from_u32_transport(code: u32) -> Self {
        match code {
            0 => Self::CytoSuccess,
            1 => Self::CytoErrNullPtr,
            2 => Self::CytoErrMmap,
            3 => Self::CytoErrInvalidLen,
            4 => Self::CytoErrAlignment,
            5 => Self::CytoErrIo,
            6 => Self::CytoErrOom,
            8 => Self::CytoErrIntegrationConfig,
            100 => Self::CytoClinicalMalignant,
            101 => Self::CytoClinicalNormal,
            7010 => Self::CytoErrMojoBadArg,
            7020 => Self::CytoErrMojoBadWork,
            7030 => Self::CytoErrMojoShape,
            7088 => Self::CytoErrMojoLibraryUnset,
            7099 => Self::CytoErrMojoUnknownTransport,
            _other => Self::CytoErrMojoUnknownTransport,
        }
    }

    /// Long-form, operator-facing explanation for dashboards and JSON tooling (**why**: compact FFI tokens from
    /// `cyto_status_message` stay stable while humans still get remediation text).
    #[must_use]
    pub fn operator_hint(self) -> &'static str {
        match self {
            Self::CytoSuccess => "completed successfully.",
            Self::CytoErrNullPtr => {
                "a required pointer was null; verify the caller passes valid session / view addresses."
            }
            Self::CytoErrMmap => {
                "memory-mapping failed; confirm the tensor path exists, is readable, and file length matches (32×D)×f32."
            }
            Self::CytoErrInvalidLen => {
                "neighbor dimensions are out of range for this build (check D ≤ CYTO_MAX_FEATURE_DIM_D and row count = 32)."
            }
            Self::CytoErrAlignment => {
                "buffer alignment violates the Mojo κ layout; align file-backed tensors to the documented stride rules."
            }
            Self::CytoErrIo => "OS I/O failed opening or reading the mmap source.",
            Self::CytoErrOom => "allocator could not reserve the session block; retry after lowering concurrency or freeing RAM.",
            Self::CytoErrIntegrationConfig => {
                "integration configuration is incomplete (for example EDGE_CYTO_DIMS_D for analyze_slide)."
            }
            Self::CytoClinicalMalignant | Self::CytoClinicalNormal => {
                "received a clinical outcome code where an error was expected; report as a bridge/version mismatch."
            }
            Self::CytoErrMojoBadArg => {
                "Mojo rejected the κ argument bundle (CytoNeighborMatrixView); confirm dims/stride/byte_length vs math_core.mojo."
            }
            Self::CytoErrMojoBadWork => {
                "Mojo reported an undersized or unusable scratch workspace; verify mojo_work_f32_elems matches invoke_mojo_curvature."
            }
            Self::CytoErrMojoShape => {
                "ring geometry did not satisfy the fixed (32×D) neighbor contract exported from coord_adapter."
            }
            Self::CytoErrMojoLibraryUnset => {
                "CYTO_MOJO_LIB_DIR is unset or this binary was built without linking libmath_core; set CYTO_MOJO_LIB_DIR to the directory containing the shared library and rebuild with the mojo_dynlib configuration."
            }
            Self::CytoErrMojoUnknownTransport => {
                "the Mojo shared object returned an unexpected status code; align math_core.mojo and libmath_core versions with this crate."
            }
        }
    }
}

/// Mirrors Mojo [`MAX_D`](mojo/math_core.mojo): integration refuses larger neighbor tensors centrally.
pub const CYTO_MAX_FEATURE_DIM_D: u32 = 256;

/// Expected byte length for a dense `(32, D)` `f32` matrix.
#[inline]
pub fn expected_matrix_byte_len(dims_d: u32) -> Option<u64> {
    let elems = (CYTO_NEIGHBOR_ROW_COUNT as u64).checked_mul(dims_d as u64)?;
    elems.checked_mul(core::mem::size_of::<f32>() as u64)
}

/// Validates that `ptr` can address `len` `f32` values read-only (why: centralize alignment rules for FFI).
///
/// # Safety
///
/// `ptr`/`len` are not read; callers must ensure dereferencing `len` elements is confined to mapped live memory elsewhere.
#[inline]
pub unsafe fn validate_f32_slice_parts(ptr: *const f32, len: usize) -> Result<(), CytoStatus> {
    if ptr.is_null() {
        return Err(CytoStatus::CytoErrNullPtr);
    }
    if len == 0 {
        return Err(CytoStatus::CytoErrInvalidLen);
    }
    let align = core::mem::align_of::<f32>();
    if !(ptr as usize).is_multiple_of(align) {
        return Err(CytoStatus::CytoErrAlignment);
    }
    Ok(())
}

/// Borrows `ptr`/`len` as a slice without allocation (why: zero-copy validation path for tests and internal checks).
///
/// # Safety
///
/// Callers uphold `slice::from_raw_parts` invariants (`ptr`, `len`, shared immutability for the returned lifetime).
pub unsafe fn as_f32_slice<'a>(ptr: *const f32, len: usize) -> Result<&'a [f32], CytoStatus> {
    validate_f32_slice_parts(ptr, len)?;
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_view_cache_line_alignment() {
        assert!(core::mem::align_of::<CytoNeighborMatrixView>() >= 64);
        assert_eq!(core::mem::align_of::<CytoNeighborMatrixView>() % 64, 0);
        assert_eq!(core::mem::size_of::<CytoNeighborMatrixView>(), 64);
        assert_eq!(
            core::mem::offset_of!(CytoNeighborMatrixView, byte_length),
            24
        );
    }

    #[test]
    fn expected_len_roundtrip() {
        let d = 128u32;
        let bl = expected_matrix_byte_len(d).unwrap();
        assert_eq!(bl, 64u64 * 128 * 4);
    }
}
