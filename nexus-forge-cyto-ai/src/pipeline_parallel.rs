//! Phase-4 batch κ driver: mmap-backed `f32` slabs + Rayon parallel cells (**gold-standard batch path**).
//!
//! **Gold-standard UI** runs per-ring κ via [`crate::geometric_orchestrator::write_enriched_geometric_output`]; this module
//! remains for **batch slide tensors** (mmap rows). Remote HTTP segmenters are **not** involved.

const _: () = assert!(core::mem::size_of::<CytoCellResult>() == 16);
const _: () = assert!(core::mem::align_of::<CytoCellResult>() == 16);

/// Per-cell explainability payload for host analytics / C interop (Phase-4 batch export).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct CytoCellResult {
    pub cell_id: u32,
    pub kappa: f32,
    pub is_malignant: u8,
    pub _pad: [u8; 7],
}

mod inner {
    use std::cell::UnsafeCell;
    use std::path::Path;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    use memmap2::Mmap;
    use rayon::prelude::*;

    use super::CytoCellResult;
    use crate::core::{
        expected_matrix_byte_len, validate_f32_slice_parts, CytoNeighborMatrixView, CytoStatus,
        CYTO_MAX_FEATURE_DIM_D, CYTO_NEIGHBOR_ROW_COUNT,
    };
    use crate::mojo_bind::{invoke_mojo_curvature, mojo_work_f32_elems};

    // SAFETY: `Vec<f32>` is owned per-thread and only mutated by that single thread.
    thread_local! {
        static MOJO_CELL_SCRATCH: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
    }

    static WARMED_DIMS_D: AtomicU32 = AtomicU32::new(0);

    pub fn warm_parallel_slide_thread_scratch(dims_d: u32) -> Result<(), CytoStatus> {
        if !(1..=CYTO_MAX_FEATURE_DIM_D).contains(&dims_d) {
            return Err(CytoStatus::CytoErrInvalidLen);
        }
        let elems = mojo_work_f32_elems(dims_d).ok_or(CytoStatus::CytoErrInvalidLen)?;
        let threads = rayon::current_num_threads().max(1);
        let iters = threads.saturating_mul(4096);
        (0..iters).into_par_iter().for_each(|_| {
            // SAFETY: each Rayon worker thread touches only its own TLS slot.
            MOJO_CELL_SCRATCH.with(|cell_ptr| unsafe {
                let v = &mut *cell_ptr.get();
                if v.len() < elems {
                    v.clear();
                    v.reserve_exact(elems);
                    v.resize(elems, 0.0);
                }
            });
        });
        WARMED_DIMS_D.store(dims_d, Ordering::Release);
        Ok(())
    }

    pub fn parallel_analyze_slide_mmap_path(
        path: &Path,
        dims_d: u32,
        optical_scale: f32,
        out: &mut [CytoCellResult],
    ) -> Result<(), CytoStatus> {
        let file = std::fs::File::open(path).map_err(|_| CytoStatus::CytoErrIo)?;
        let mmap = unsafe { Mmap::map(&file).map_err(|_| CytoStatus::CytoErrMmap)? };
        parallel_analyze_slide_mmap(&mmap, dims_d, optical_scale, out)
    }

    pub fn parallel_analyze_slide_mmap(
        mmap: &Mmap,
        dims_d: u32,
        optical_scale: f32,
        out: &mut [CytoCellResult],
    ) -> Result<(), CytoStatus> {
        let byte_len = mmap.len();
        if !byte_len.is_multiple_of(core::mem::size_of::<f32>()) {
            return Err(CytoStatus::CytoErrInvalidLen);
        }
        let floats = byte_len / core::mem::size_of::<f32>();
        let base = mmap.as_ptr() as *const f32;
        let sl = unsafe { core::slice::from_raw_parts(base, floats) };
        parallel_analyze_slide(sl, dims_d, optical_scale, out)
    }

    pub fn parallel_analyze_slide(
        mmap_f32: &[f32],
        dims_d: u32,
        optical_scale: f32,
        out: &mut [CytoCellResult],
    ) -> Result<(), CytoStatus> {
        if WARMED_DIMS_D.load(Ordering::Acquire) != dims_d {
            warm_parallel_slide_thread_scratch(dims_d)?;
        }
        if !(1..=CYTO_MAX_FEATURE_DIM_D).contains(&dims_d) {
            return Err(CytoStatus::CytoErrInvalidLen);
        }

        let cell_elems_usize = CYTO_NEIGHBOR_ROW_COUNT
            .checked_mul(dims_d)
            .and_then(|e| usize::try_from(e).ok())
            .ok_or(CytoStatus::CytoErrInvalidLen)?;
        let scratch_elems = mojo_work_f32_elems(dims_d).ok_or(CytoStatus::CytoErrInvalidLen)?;
        let cell_byte_len =
            expected_matrix_byte_len(dims_d).ok_or(CytoStatus::CytoErrInvalidLen)?;

        let ncells = out.len();
        let required_floats = ncells
            .checked_mul(cell_elems_usize)
            .ok_or(CytoStatus::CytoErrInvalidLen)?;
        if mmap_f32.len() != required_floats {
            return Err(CytoStatus::CytoErrInvalidLen);
        }

        if !mmap_f32.is_empty() {
            let base_ptr = mmap_f32.as_ptr();
            unsafe {
                validate_f32_slice_parts(base_ptr, mmap_f32.len())?;
            }
        }

        // Per-invocation error flag (Arc<AtomicU32>) so concurrent calls to
        // parallel_analyze_slide do not share a single static error flag.
        let batch_err = Arc::new(AtomicU32::new(0));

        mmap_f32
            .par_chunks_exact(cell_elems_usize)
            .zip(out.par_iter_mut())
            .enumerate()
            .for_each(|(cell_index, (matrix_f32, dst))| {
                if batch_err.load(Ordering::Relaxed) != 0 {
                    return;
                }

                let cell_id = match u32::try_from(cell_index) {
                    Ok(v) => v,
                    Err(_) => {
                        let _ = batch_err.compare_exchange(
                            0,
                            CytoStatus::CytoErrInvalidLen as u32,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        );
                        return;
                    }
                };

                // SAFETY: TLS slot is owned by this Rayon worker thread alone;
                // we take a single &mut for the duration of the Mojo call with no
                // nested TLS access, so reentrancy cannot occur (the prior
                // RefCell::borrow_mut could panic on reentry; UnsafeCell removes
                // that panic surface and we enforce single-borrow by control flow).
                MOJO_CELL_SCRATCH.with(|cell_ptr| unsafe {
                    let scratch: &mut [f32] = {
                        let v: &mut Vec<f32> = &mut *cell_ptr.get();
                        if v.len() < scratch_elems {
                            v.clear();
                            v.reserve_exact(scratch_elems);
                            v.resize(scratch_elems, 0.0);
                        }
                        &mut v[..scratch_elems]
                    };

                    let data_ptr = matrix_f32.as_ptr();
                    if let Err(st) = validate_f32_slice_parts(data_ptr, cell_elems_usize)
                    {
                        let _ = batch_err.compare_exchange(
                            0,
                            st as u32,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        );
                        return;
                    }

                    let view = CytoNeighborMatrixView {
                        data: data_ptr,
                        dims_d,
                        row_count: CYTO_NEIGHBOR_ROW_COUNT,
                        stride_elems: dims_d,
                        _implicit_pad_before_byte_length: 0,
                        byte_length: cell_byte_len,
                        _align_reserved_tail: [0_u8; 32],
                    };

                    let mut kappa: f32 = 0.0;
                    let mut clinical_raw: u32 = 0;

                    let mcode = invoke_mojo_curvature(
                        core::ptr::addr_of!(view),
                        optical_scale,
                        core::ptr::addr_of_mut!(kappa),
                        core::ptr::addr_of_mut!(clinical_raw),
                        scratch.as_mut_ptr(),
                        scratch_elems,
                    );
                    let mstat = CytoStatus::from_u32_transport(mcode);
                    if mstat != CytoStatus::CytoSuccess {
                        let _ = batch_err.compare_exchange(
                            0,
                            mstat as u32,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        );
                        return;
                    }

                    let clinical = CytoStatus::from_u32_transport(clinical_raw);
                    let mal = match clinical {
                        CytoStatus::CytoClinicalMalignant => 1_u8,
                        CytoStatus::CytoClinicalNormal => 0_u8,
                        _ => {
                            let _ = batch_err.compare_exchange(
                                0,
                                CytoStatus::CytoErrMojoUnknownTransport as u32,
                                Ordering::AcqRel,
                                Ordering::Relaxed,
                            );
                            return;
                        }
                    };

                    *dst = CytoCellResult {
                        cell_id,
                        kappa,
                        is_malignant: mal,
                        _pad: [0_u8; 7],
                    };
                });
            });

        let err = batch_err.load(Ordering::Acquire);
        if err == 0 {
            Ok(())
        } else {
            Err(CytoStatus::from_u32_transport(err))
        }
    }
}

pub use inner::{
    parallel_analyze_slide, parallel_analyze_slide_mmap, parallel_analyze_slide_mmap_path,
    warm_parallel_slide_thread_scratch,
};
