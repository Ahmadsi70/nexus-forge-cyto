//! Geometric smoke tests: deterministic `(32, 2)` tensors through mmap + Rust κ core.

use std::ffi::CString;

use std::fs;
use std::io::Write;
use std::mem::size_of;

use nexus_forge_cyto::{
    analyze_slide_with_dims, expected_matrix_byte_len, CytoNeighborSessionGuard, CytoStatus,
    CYTO_NEIGHBOR_ROW_COUNT,
};

const D: u32 = 2;
const ROWS: usize = CYTO_NEIGHBOR_ROW_COUNT as usize;
const FLOATS: usize = ROWS * D as usize;
const BYTES: usize = FLOATS * size_of::<f32>();

/// Shape A: 32 samples on a perfect unit circle in feature space (cos θ, sin θ), θ = 2πr/32.
fn generate_circle_32x2() -> Vec<f32> {
    let mut v = Vec::with_capacity(FLOATS);
    let n = ROWS as f32;
    for r in 0..ROWS {
        let theta = std::f32::consts::TAU * (r as f32) / n;
        v.push(theta.cos());
        v.push(theta.sin());
    }
    debug_assert_eq!(v.len(), FLOATS);
    v
}

/// Shape B: highly anisotropic “spiked star” — strong axis stretch plus periodic radial spikes.
fn generate_spiked_star_32x2() -> Vec<f32> {
    let mut v = Vec::with_capacity(FLOATS);
    let n = ROWS as f32;
    for r in 0..ROWS {
        let theta = std::f32::consts::TAU * (r as f32) / n;
        let mut x = 18.0_f32 * theta.cos();
        let mut y = 0.35_f32 * theta.sin();
        if r % 4 == 0 {
            x += 9.0_f32;
        }
        if r % 8 == 0 {
            y -= 6.5_f32;
        }
        v.push(x);
        v.push(y);
    }
    debug_assert_eq!(v.len(), FLOATS);
    v
}

fn write_wsi_mock(path: &std::path::Path, floats: &[f32]) {
    assert_eq!(
        floats.len(),
        FLOATS,
        "must be exactly one (32, 2) dense matrix"
    );
    let expected = expected_matrix_byte_len(D).expect("valid D") as usize;
    assert_eq!(
        BYTES, expected as usize,
        "float count must match Phase-1 FFI `expected_matrix_byte_len`"
    );
    let mut f = fs::File::create(path).expect("create temp WSI");
    for &x in floats {
        f.write_all(&x.to_ne_bytes()).expect("write f32");
    }
    f.flush().expect("flush");
    let len = fs::metadata(path).expect("metadata").len() as usize;
    assert_eq!(len, BYTES, "on-disk byte length must equal 32×2×4");
}

/// `CytoNeighborSessionGuard`: construct from mmap-backed file and drop cleanly (Phase-3 RAII).
#[test]
fn geometry_session_guard_drop_no_fault() {
    let circle = generate_circle_32x2();
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    write_wsi_mock(tmp.path(), &circle);

    let c_path = CString::new(tmp.path().to_str().unwrap()).expect("cstring path");
    let guard = CytoNeighborSessionGuard::try_open_legacy(c_path.as_c_str(), D)
        .expect("mmap session opens for dense (32,2) file");
    let _ptr = guard.as_ptr();
    std::hint::black_box(_ptr);
    drop(guard);
}

#[test]
fn geometry_tempfile_layout_and_pipeline_invocation_smoke() {
    let circle = generate_circle_32x2();
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    write_wsi_mock(tmp.path(), &circle);

    let res = analyze_slide_with_dims(tmp.path().to_str().unwrap(), D, 1.0_f32);
    match res {
        Ok((kappa, clinical)) => {
            assert!(kappa.is_finite(), "κ must be finite when engine returns Ok");
            assert!(
                matches!(
                    clinical,
                    CytoStatus::CytoClinicalMalignant | CytoStatus::CytoClinicalNormal
                ),
                "unexpected clinical transport: {:?}",
                clinical
            );
        }
        Err(e) => panic!("unexpected pipeline error: {:?}", e),
    }
}

/// κ discriminability in `D==2`: isotropic manifold → κ≈½; anisotropic “spiked” geometry → collapsed κ (`<~0.35`).
#[test]
fn geometry_kappa_baseline_circle_vs_spiked_star() {
    let tmp_a = tempfile::NamedTempFile::new().expect("tempfile A");
    let tmp_b = tempfile::NamedTempFile::new().expect("tempfile B");
    write_wsi_mock(tmp_a.path(), &generate_circle_32x2());
    write_wsi_mock(tmp_b.path(), &generate_spiked_star_32x2());

    let circle = analyze_slide_with_dims(tmp_a.path().to_str().unwrap(), D, 1.0_f32)
        .expect("rust κ engine on circle manifold");
    let star = analyze_slide_with_dims(tmp_b.path().to_str().unwrap(), D, 1.0_f32)
        .expect("rust κ engine on spiked star");
    let (kappa_a, circle_clin) = circle;
    let (kappa_b, star_clin) = star;

    eprintln!("κ (optical_scale=1.0): circle={kappa_a:.9}, spiked_star={kappa_b:.9}");

    assert_eq!(
        circle_clin,
        CytoStatus::CytoClinicalNormal,
        "isotropic manifold should classify as NORMAL (high κ≈0.5 > mid-band gate)"
    );
    assert_eq!(
        star_clin,
        CytoStatus::CytoClinicalMalignant,
        "anisotropic morphology should classify as MALIGNANT (collapsed κ)"
    );

    assert!(
        kappa_a > 0.45,
        "Shape A (circle) expected balanced 2D spectrum → κ>0.45; got {kappa_a}"
    );
    assert!(
        kappa_b < 0.35,
        "Shape B (spiked star) expected anisotropic collapse → κ<0.35; got {kappa_b}"
    );
    assert!(
        kappa_b < kappa_a - 0.05,
        "discriminative gap: κ_star ({kappa_b}) should sit below κ_circle ({kappa_a})"
    );
}
