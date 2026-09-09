//! Memory-mapped WSI/tensor backing stores with explicit length checks (`memmap2`).
//!
//! **Why this module exists**: Centralizes file policy (read-only, shared map) so FFI stays a thin coercion layer without duplicating mapped-file error handling.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::core::{
    expected_matrix_byte_len, CytoNeighborMatrixView, CytoStatus, CYTO_NEIGHBOR_ROW_COUNT,
};

/// Internal session owning the mmap and logical `(32, D)` shape.
///
/// **Why opaque to C callers**: preserves Rust drop order for `Mmap` while exporting only raw pointers externally.
pub struct MappedNeighborMatrixSession {
    mmap: Mmap,
    dims_d: u32,
}

impl MappedNeighborMatrixSession {
    /// Maps `path` read-only and pins the `(CYTO_NEIGHBOR_ROW_COUNT, dims_d)` `f32` matrix view.
    ///
    /// **Why conservative IO errors**: Distinguishes mapping failures from shape violations for operator telemetry.
    pub fn open(path: &Path, dims_d: u32) -> Result<Self, CytoStatus> {
        if dims_d == 0 {
            return Err(CytoStatus::CytoErrInvalidLen);
        }
        let Some(expected) = expected_matrix_byte_len(dims_d) else {
            return Err(CytoStatus::CytoErrInvalidLen);
        };
        let file = File::open(path).map_err(|_| CytoStatus::CytoErrIo)?;
        // Safety: documented by `memmap2` — no mutable aliases; backing file stays open via `mmap` lifetime rules.
        let mmap = unsafe { Mmap::map(&file).map_err(|_| CytoStatus::CytoErrMmap)? };
        let len = mmap.len() as u64;
        if len < expected {
            return Err(CytoStatus::CytoErrInvalidLen);
        }

        Ok(Self { mmap, dims_d })
    }

    /// Zero-copy typed view over mapped bytes (panic-free; returns status for bad alignment).
    pub fn neighbor_view(&self) -> Result<CytoNeighborMatrixView, CytoStatus> {
        let base = self.mmap.as_ptr() as *const f32;
        let total = expected_matrix_elem_count(self.dims_d)?;
        unsafe { crate::core::validate_f32_slice_parts(base, total)? };

        Ok(CytoNeighborMatrixView {
            data: base,
            dims_d: self.dims_d,
            row_count: CYTO_NEIGHBOR_ROW_COUNT,
            stride_elems: self.dims_d,
            _implicit_pad_before_byte_length: 0,
            byte_length: self.mmap.len() as u64,
            _align_reserved_tail: [0_u8; 32],
        })
    }
}

#[inline]
fn expected_matrix_elem_count(dims_d: u32) -> Result<usize, CytoStatus> {
    let rows = CYTO_NEIGHBOR_ROW_COUNT as usize;
    let cols = dims_d as usize;
    rows.checked_mul(cols).ok_or(CytoStatus::CytoErrInvalidLen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn maps_minimal_dense_file() {
        let d: u32 = 4;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let elems = (CYTO_NEIGHBOR_ROW_COUNT as usize) * (d as usize);
        for i in 0..elems {
            let f = i as f32;
            tmp.write_all(&f.to_ne_bytes()).unwrap();
        }
        tmp.flush().unwrap();

        let s = MappedNeighborMatrixSession::open(tmp.path(), d).expect("open");
        let v = s.neighbor_view().expect("view");
        assert_eq!(v.row_count, CYTO_NEIGHBOR_ROW_COUNT);
        assert_eq!(v.dims_d, d);
        unsafe {
            let sl = crate::core::as_f32_slice(v.data, elems).unwrap();
            assert_eq!(sl[0], 0.0);
        }
    }
}
