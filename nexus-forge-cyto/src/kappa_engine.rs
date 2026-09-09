//! Phase-2 κ (curvature) core — faithful Rust port of `mojo/math_core.mojo`.
//!
//! **Why pure Rust**: identical `(32, D)` neighbor contract on Windows and Linux without
//! `libmath_core.so` / Modular SDK; numeric path mirrors Mojo constants and control flow.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::approx_constant
)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
};

use crate::core::{CytoNeighborMatrixView, CytoStatus, CYTO_MAX_FEATURE_DIM_D, CYTO_NEIGHBOR_ROW_COUNT};

const N_ROWS: usize = 64;
const FREQ_KEEP: u32 = 16;
const SAFE_TRACE_EPS: f32 = 1e-7;
const POWER_ITER_NORM_EPS: f32 = 1e-7;
const KAPPA_NONFINITE_SENTINEL: f32 = 0.5;
const POWER_ITER_MAX: u32 = 64;
const POWER_ITER_TOL_LAMBDA: f32 = 1e-5;
const BASE_KAPPA_THRESHOLD: f32 = 0.40;
const FFT_PI: f32 = 3.14159265358979323846264338;

const CYTO_FFI_OK: u32 = 0;
const CYTO_FFI_BAD_ARG: u32 = 7010;
const CYTO_FFI_BAD_WORK: u32 = 7020;
const CYTO_FFI_SHAPE: u32 = 7030;

/// Scratch `f32` word count: `32*D + 64 + D² + 2*D` (matches Mojo `math_core.mojo`).
#[inline]
pub fn kappa_work_f32_elems(dims_d: u32) -> Option<usize> {
    let rows: usize = N_ROWS;
    let d = dims_d as usize;
    let first = rows.checked_mul(d)?;
    let fft = rows.checked_mul(2)?;
    let cov = d.checked_mul(d)?;
    let tails = d.checked_mul(2)?;
    first.checked_add(fft)?.checked_add(cov)?.checked_add(tails)
}

#[inline]
fn bit_reverse_64(i: u32) -> u32 {
    let mut x = i;
    let mut rv = 0_u32;
    for _ in 0..6 {
        rv = (rv << 1) | (x & 1);
        x >>= 1;
    }
    rv
}

fn fft_inplace64_complex(re: &mut [f32], im: &mut [f32], forward: bool) {
    for bi in 0..N_ROWS {
        let bj = bit_reverse_64(bi as u32) as usize;
        if bj > bi {
            re.swap(bi, bj);
            im.swap(bi, bj);
        }
    }

    let mut stage = 1_u32;
    while stage <= 6 {
        let m = 1_u32 << stage;
        let half = m >> 1;
        let mut grp = 0_u32;
        while grp < N_ROWS as u32 {
            let mut jj = 0_u32;
            while jj < half {
                let k0 = (grp + jj) as usize;
                let k1 = (grp + jj + half) as usize;
                let base = 2.0_f32 * FFT_PI * jj as f32 / m as f32;
                let angle = if forward { -base } else { base };
                let wr = angle.cos();
                let wi = angle.sin();
                let r1 = re[k1];
                let i1 = im[k1];
                let tr = r1 * wr - i1 * wi;
                let ti_m = r1 * wi + i1 * wr;
                let r0 = re[k0];
                let i0 = im[k0];
                re[k1] = r0 - tr;
                im[k1] = i0 - ti_m;
                re[k0] = r0 + tr;
                im[k0] = i0 + ti_m;
                jj += 1;
            }
            grp += m;
        }
        stage += 1;
    }

    if forward {
        return;
    }

    let inv_scale = 1.0_f32 / N_ROWS as f32;
    for ix in 0..N_ROWS {
        re[ix] *= inv_scale;
        im[ix] *= inv_scale;
    }
}

fn hermitian_mask64(re: &mut [f32], im: &mut [f32]) {
    let mut kk = FREQ_KEEP;
    let stop = N_ROWS as u32 - FREQ_KEEP;
    while kk <= stop {
        let ix = kk as usize;
        if kk != 1 && kk != (N_ROWS as u32 - 1) {
            let k_conj = N_ROWS as u32 - kk;
            let jx = k_conj as usize;
            re[ix] = 0.0;
            im[ix] = 0.0;
            re[jx] = 0.0;
            im[jx] = 0.0;
        }
        kk += 1;
    }
}

fn smooth_neighbor_column_into_dense(
    base: &[f32],
    stride: usize,
    col: usize,
    column_mean_raw: f32,
    dense: &mut [f32],
    dense_stride: usize,
    fft_re: &mut [f32],
    fft_im: &mut [f32],
) {
    for r_src in 0..N_ROWS {
        let src_i = r_src * stride + col;
        fft_re[r_src] = base[src_i] - column_mean_raw;
        fft_im[r_src] = 0.0;
    }
    fft_inplace64_complex(fft_re, fft_im, true);
    hermitian_mask64(fft_re, fft_im);
    fft_inplace64_complex(fft_re, fft_im, false);
    for r_src in 0..N_ROWS {
        let dst_i = r_src * dense_stride + col;
        dense[dst_i] = fft_re[r_src];
    }
}

// ---------------------------------------------------------------------------
// dot_dense: AVX2+FMA SIMD with scalar fallback (stable Rust, no deps)
// ---------------------------------------------------------------------------

/// Scalar fallback — always safe, works everywhere.
#[inline]
fn dot_dense_scalar(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut scalar = 0.0_f32;
    for j in 0..dim {
        scalar += a[j] * b[j];
    }
    scalar
}

/// AVX2+FMA accelerated dot product (8 × f32 per iteration).
///
/// # Safety
/// Caller must guarantee the CPU supports AVX2 and FMA (checked by
/// `is_x86_feature_detected!` before calling).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_dense_avx2_fma(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut j = 0usize;

    // Main loop: 8-lane FMA accumulation
    let simd_end = dim - (dim % 8);
    while j < simd_end {
        let a8 = _mm256_loadu_ps(a.as_ptr().add(j));
        let b8 = _mm256_loadu_ps(b.as_ptr().add(j));
        acc = _mm256_fmadd_ps(a8, b8, acc);
        j += 8;
    }

    // Horizontal sum of the 8-lane accumulator into one f32
    //   lo128 = lower 4 lanes,  hi128 = upper 4 lanes
    let lo128 = _mm256_extractf128_ps::<0>(acc); // [a0, a1, a2, a3]
    let hi128 = _mm256_extractf128_ps::<1>(acc); // [a4, a5, a6, a7]
    let sum128 = _mm_add_ps(lo128, hi128); // [a0+a4, a1+a5, a2+a6, a3+a7]
    let hadd = _mm_hadd_ps(sum128, sum128); // [a0+a4+a1+a5, a2+a6+a3+a7, ...]
    let hadd = _mm_hadd_ps(hadd, hadd); // all 4 lanes = total
    let mut result = _mm_cvtss_f32(hadd);

    // Scalar remainder (at most 7 elements)
    while j < dim {
        result += a[j] * b[j];
        j += 1;
    }

    result
}

/// Public entry point: dispatches at runtime to the best implementation.
#[inline]
pub fn dot_dense(a: &[f32], b: &[f32], dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_dense_avx2_fma(a, b, dim) };
        }
    }
    dot_dense_scalar(a, b, dim)
}

// ---------------------------------------------------------------------------
// axpy_row: AVX2+FMA SIMD with scalar fallback (stable Rust, no deps)
// ---------------------------------------------------------------------------

/// Scalar fallback.
#[inline]
fn axpy_row_scalar(acc_row: &mut [f32], scale: f32, x_row: &[f32], dim: usize) {
    for j in 0..dim {
        acc_row[j] += scale * x_row[j];
    }
}

/// AVX2+FMA accelerated AXPY: `acc_row += scale * x_row` (8-lane FMA).
///
/// # Safety
/// Caller must guarantee the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn axpy_row_avx2_fma(acc_row: &mut [f32], scale: f32, x_row: &[f32], dim: usize) {
    let scale_vec = _mm256_set1_ps(scale);
    let mut j = 0usize;

    let simd_end = dim - (dim % 8);
    while j < simd_end {
        let acc8 = _mm256_loadu_ps(acc_row.as_ptr().add(j));
        let x8 = _mm256_loadu_ps(x_row.as_ptr().add(j));
        let result = _mm256_fmadd_ps(scale_vec, x8, acc8);
        _mm256_storeu_ps(acc_row.as_mut_ptr().add(j), result);
        j += 8;
    }

    // Scalar remainder
    while j < dim {
        acc_row[j] += scale * x_row[j];
        j += 1;
    }
}

/// Public entry point: dispatches at runtime to the best implementation.
#[inline]
pub fn axpy_row(acc_row: &mut [f32], scale: f32, x_row: &[f32], dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { axpy_row_avx2_fma(acc_row, scale, x_row, dim) };
            return;
        }
    }
    axpy_row_scalar(acc_row, scale, x_row, dim);
}

#[inline]
fn f32_max(a: f32, b: f32) -> f32 {
    if a >= b { a } else { b }
}

#[inline]
fn f32_min(a: f32, b: f32) -> f32 {
    if a <= b { a } else { b }
}

#[inline]
fn kappa_safety_guard(x: f32) -> f32 {
    if x.is_nan() {
        return KAPPA_NONFINITE_SENTINEL;
    }
    let mut ax = x;
    if ax < 0.0 {
        ax = -ax;
    }
    if ax >= 3.402823e38_f32 {
        return KAPPA_NONFINITE_SENTINEL;
    }
    f32_min(f32_max(x, 0.0), 1.0)
}

#[inline]
fn fma_max_abs(a: f32, b: f32) -> f32 {
    let d = a - b;
    if d >= 0.0 { d } else { -d }
}

/// Port of `@export("cyto_mojo_phase2_curvature_core")` — same status codes and layout rules.
///
/// **Safety**: `view.data` must address `row_count × stride_elems` valid `f32` for the call duration.
pub unsafe fn phase2_curvature_core(
    view: &CytoNeighborMatrixView,
    optical_scale: f32,
    out_kappa: *mut f32,
    out_clinical: *mut u32,
    work: *mut f32,
    work_elems: u32,
) -> u32 {
    if view.data.is_null() || out_kappa.is_null() || out_clinical.is_null() || work.is_null() {
        return CYTO_FFI_BAD_ARG;
    }

    let d = view.dims_d as usize;
    if d == 0 || view.dims_d > CYTO_MAX_FEATURE_DIM_D {
        return CYTO_FFI_SHAPE;
    }
    if view.row_count != CYTO_NEIGHBOR_ROW_COUNT {
        return CYTO_FFI_SHAPE;
    }
    let stride = view.stride_elems as usize;
    if stride < d {
        return CYTO_FFI_SHAPE;
    }

    let need = N_ROWS * d + 2 * N_ROWS + d * d + 2 * d;
    if (work_elems as usize) < need {
        return CYTO_FFI_BAD_WORK;
    }

    let data_len = N_ROWS * stride;
    let data = core::slice::from_raw_parts(view.data, data_len);
    let work_sl = core::slice::from_raw_parts_mut(work, work_elems as usize);

    let sn = N_ROWS * d;
    let (smooth, rest) = work_sl.split_at_mut(sn);
    let (fft_re, rest) = rest.split_at_mut(N_ROWS);
    let (fft_im, rest) = rest.split_at_mut(N_ROWS);
    let (cov, rest) = rest.split_at_mut(d * d);
    let (vec_v, vec_w) = rest.split_at_mut(d);

    cov.fill(0.0);
    for r in 0..N_ROWS {
        for dc in 0..d {
            let imu = r * stride + dc;
            cov[dc] += data[imu];
        }
    }
    let inv_nr_mu = 1.0_f32 / N_ROWS as f32;
    for cmd in 0..d {
        cov[cmd] *= inv_nr_mu;
    }

    for col in 0..d {
        smooth_neighbor_column_into_dense(
            data,
            stride,
            col,
            cov[col],
            smooth,
            d,
            &mut fft_re[..N_ROWS],
            &mut fft_im[..N_ROWS],
        );
    }

    vec_w.fill(0.0);
    for rr in 0..N_ROWS {
        for cc in 0..d {
            let rr_i = rr * d + cc;
            vec_w[cc] += smooth[rr_i];
        }
    }
    let inv_nr = 1.0_f32 / N_ROWS as f32;
    for cm in 0..d {
        vec_w[cm] *= inv_nr;
    }

    cov.fill(0.0);
    let mut trace_acc = 0.0_f32;
    for row_r in 0..N_ROWS {
        for cc in 0..d {
            let ij = row_r * d + cc;
            smooth[ij] -= vec_w[cc];
        }
        let row_begin = row_r * d;
        let row_slice = &smooth[row_begin..row_begin + d];
        trace_acc += dot_dense(row_slice, row_slice, d);
        for ci in 0..d {
            let scale = smooth[row_r * d + ci];
            let crow = &mut cov[ci * d..(ci + 1) * d];
            axpy_row(crow, scale, row_slice, d);
        }
    }

    let inv_nm1 = 1.0_f32 / (N_ROWS as f32 - 1.0);
    for zp in 0..(d * d) {
        cov[zp] *= inv_nm1;
    }

    let trace = trace_acc * inv_nm1;
    if trace <= 0.0 {
        unsafe {
            *out_kappa = 0.0;
            *out_clinical = CytoStatus::CytoClinicalMalignant as u32;
        }
        return CYTO_FFI_OK;
    }

    let safe_trace = f32_max(trace, SAFE_TRACE_EPS);

    let ratio = if d == 2 {
        let cxx = cov[0];
        let cxy = cov[1];
        let cyy = cov[3];
        let mut disc_inner = (cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy;
        if disc_inner < 0.0 {
            disc_inner = 0.0;
        }
        let eig1 = 0.5 * ((cxx + cyy) + disc_inner.sqrt());
        eig1 / safe_trace
    } else {
        vec_v.fill(0.0);
        let init_scale = 1.0 / (d as f32).sqrt();
        for vi in 0..d {
            vec_v[vi] = init_scale;
            vec_w[vi] = 0.0;
        }
        let mut eigen_est = 0.0_f32;
        let mut eigen_prev = -1.0_f32;
        let mut it = 0_u32;
        while it < POWER_ITER_MAX {
            for ri in 0..d {
                let rowp = &cov[ri * d..(ri + 1) * d];
                vec_w[ri] = dot_dense(rowp, vec_v, d);
            }
            eigen_est = dot_dense(vec_v, vec_w, d);
            let norm_w = dot_dense(vec_w, vec_w, d).sqrt();
            if norm_w <= POWER_ITER_NORM_EPS {
                eigen_est = 0.0;
                break;
            }
            let inv_norm = 1.0 / norm_w;
            for xi in 0..d {
                vec_v[xi] = vec_w[xi] * inv_norm;
            }
            if fma_max_abs(eigen_est, eigen_prev) < POWER_ITER_TOL_LAMBDA {
                break;
            }
            eigen_prev = eigen_est;
            it += 1;
        }
        eigen_est / safe_trace
    };

    let mut ratio_clamped = ratio;
    if ratio_clamped > 1.0 {
        ratio_clamped = 1.0;
    }
    let kappa = kappa_safety_guard(1.0 - ratio_clamped);
    let thr = BASE_KAPPA_THRESHOLD * optical_scale;
    let decision = if kappa < thr {
        CytoStatus::CytoClinicalMalignant as u32
    } else {
        CytoStatus::CytoClinicalNormal as u32
    };

    unsafe {
        *out_kappa = kappa;
        *out_clinical = decision;
    }
    CYTO_FFI_OK
}

/// Rust κ entry used by pipeline/orchestrator (replaces Mojo FFI when library unset).
#[inline]
pub unsafe fn invoke_kappa_curvature(
    view: *const CytoNeighborMatrixView,
    optical_scale: f32,
    out_kappa: *mut f32,
    out_clinical: *mut u32,
    scratch: *mut f32,
    scratch_elems: usize,
) -> u32 {
    if view.is_null() {
        return CytoStatus::CytoErrNullPtr as u32;
    }
    let work_elems = match u32::try_from(scratch_elems) {
        Ok(n) => n,
        Err(_) => return CytoStatus::CytoErrInvalidLen as u32,
    };
    unsafe {
        phase2_curvature_core(
            &*view,
            optical_scale,
            out_kappa,
            out_clinical,
            scratch,
            work_elems,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::expected_matrix_byte_len;

    fn matrix_view_from(data: &mut [f32], dims_d: u32) -> CytoNeighborMatrixView {
        CytoNeighborMatrixView {
            data: data.as_ptr(),
            dims_d,
            row_count: CYTO_NEIGHBOR_ROW_COUNT,
            stride_elems: dims_d,
            _implicit_pad_before_byte_length: 0,
            byte_length: expected_matrix_byte_len(dims_d).unwrap(),
            _align_reserved_tail: [0_u8; 32],
        }
    }

    fn generate_circle_64x2() -> Vec<f32> {
        let mut v = Vec::with_capacity(N_ROWS * 2);
        let n = N_ROWS as f32;
        for r in 0..N_ROWS {
            let theta = std::f32::consts::TAU * (r as f32) / n;
            v.push(theta.cos());
            v.push(theta.sin());
        }
        v
    }

    fn generate_spiked_star_64x2() -> Vec<f32> {
        let mut v = Vec::with_capacity(N_ROWS * 2);
        let n = N_ROWS as f32;
        for r in 0..N_ROWS {
            let theta = std::f32::consts::TAU * (r as f32) / n;
            let mut x = 18.0_f32 * theta.cos();
            let mut y = 0.35_f32 * theta.sin();
            if r % 4 == 0 {
                x += 9.0;
            }
            if r % 8 == 0 {
                y -= 6.5;
            }
            v.push(x);
            v.push(y);
        }
        v
    }

    fn run_kappa(data: &mut [f32]) -> (f32, CytoStatus) {
        let d = 2_u32;
        let scratch_n = kappa_work_f32_elems(d).unwrap();
        let mut scratch = vec![0.0_f32; scratch_n];
        let view = matrix_view_from(data, d);
        let mut kappa = 0.0_f32;
        let mut clinical_raw = 0_u32;
        let code = unsafe {
            phase2_curvature_core(
                &view,
                1.0,
                std::ptr::addr_of_mut!(kappa),
                std::ptr::addr_of_mut!(clinical_raw),
                scratch.as_mut_ptr(),
                scratch.len() as u32,
            )
        };
        assert_eq!(code, CYTO_FFI_OK);
        (kappa, CytoStatus::from_u32_transport(clinical_raw))
    }

    #[test]
    fn kappa_baseline_circle_vs_spiked_star() {
        let mut circle = generate_circle_64x2();
        let mut star = generate_spiked_star_64x2();
        let (kappa_a, circle_clin) = run_kappa(&mut circle);
        let (kappa_b, star_clin) = run_kappa(&mut star);

        assert_eq!(circle_clin, CytoStatus::CytoClinicalNormal);
        assert_eq!(star_clin, CytoStatus::CytoClinicalMalignant);
        assert!(kappa_a > 0.45, "circle κ={kappa_a}");
        assert!(kappa_b < 0.35, "star κ={kappa_b}");
        assert!(kappa_b < kappa_a - 0.05);
    }

    #[test]
    fn kappa_work_elems_matches_mojo_formula() {
        assert_eq!(kappa_work_f32_elems(2), Some(64 * 2 + 128 + 4 + 4));
    }
}
