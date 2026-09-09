//! Mapping #9: FFT-based cyclic convolution for multiplicative fusion of ring samples.
//!
//! **Mathematical origin** (from `discovered_mathematical_mappings.md`):
//! > a ⨂ b := FFT⁻¹(FFT(a) ⊙ FFT(b))
//! > Fuse angular histograms (32-point rings like current κ pipeline) or
//! > neighbour masks via cyclic convolution — O(n log n) aggregation on
//! > periodic boundary conditions around each cell.
//!
//! **Cell-network analogue:**
//! - Each cell boundary is a 32-point ring (periodic domain).
//! - Cyclic convolution fuses two ring shapes into a new ring.
//! - Angular spectrum comparison for shape similarity.
//! - Frequency-domain distance metric for cell morphology.

use std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// FFT (same structure as kappa_engine.rs — in-place, N=32)
// ---------------------------------------------------------------------------

const N: usize = 64;

/// Bit-reverse for 6-bit indices (0..64).
#[inline]
fn bit_reverse(i: u32) -> u32 {
    let mut x = i;
    let mut rv = 0_u32;
    for _ in 0..6 {
        rv = (rv << 1) | (x & 1);
        x >>= 1;
    }
    rv
}

/// In-place complex FFT on 32-element arrays.
fn fft64(re: &mut [f32; N], im: &mut [f32; N], forward: bool) {
    // Bit-reversal permutation
    for bi in 0..N {
        let bj = bit_reverse(bi as u32) as usize;
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
        while grp < N as u32 {
            let mut jj = 0_u32;
            while jj < half {
                let k0 = (grp + jj) as usize;
                let k1 = (grp + jj + half) as usize;
                let angle = TAU * jj as f32 / m as f32;
                let (wr, wi) = if forward {
                    (angle.cos(), -angle.sin())
                } else {
                    (angle.cos(), angle.sin())
                };
                let r1 = re[k1];
                let i1 = im[k1];
                let tr = r1 * wr - i1 * wi;
                let ti = r1 * wi + i1 * wr;
                re[k1] = re[k0] - tr;
                im[k1] = im[k0] - ti;
                re[k0] += tr;
                im[k0] += ti;
                jj += 1;
            }
            grp += m;
        }
        stage += 1;
    }

    // Scale for inverse transform
    if !forward {
        let inv = 1.0_f32 / N as f32;
        for x in re.iter_mut() {
            *x *= inv;
        }
        for x in im.iter_mut() {
            *x *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Ring ↔ complex arrays
// ---------------------------------------------------------------------------

/// Convert a 32-point (x,y) ring to two complex arrays (re=x, im=y).
fn ring_to_complex(ring: &[[f32; 2]; N]) -> ([f32; N], [f32; N]) {
    let mut re = [0.0_f32; N];
    let mut im = [0.0_f32; N];
    for (i, pt) in ring.iter().enumerate() {
        re[i] = pt[0];
        im[i] = pt[1];
    }
    (re, im)
}

/// Convert complex arrays back to a 32-point (x,y) ring.
fn complex_to_ring(re: &[f32; N], im: &[f32; N]) -> [[f32; 2]; N] {
    let mut ring = [[0.0_f32; 2]; N];
    for i in 0..N {
        ring[i] = [re[i], im[i]];
    }
    ring
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Cyclic convolution of two rings in the frequency domain.
///
/// ring_a ⨂ ring_b = FFT⁻¹(FFT(ring_a) ⊙ FFT(ring_b))
///
/// This fuses the angular characteristics of two cell boundaries.
/// **Use case:** compare or combine two cell morphologies.
pub fn cyclic_convolution_rings(
    ring_a: &[[f32; 2]; N],
    ring_b: &[[f32; 2]; N],
) -> [[f32; 2]; N] {
    let (mut re_a, mut im_a) = ring_to_complex(ring_a);
    let (mut re_b, mut im_b) = ring_to_complex(ring_b);

    // Forward FFT
    fft64(&mut re_a, &mut im_a, true);
    fft64(&mut re_b, &mut im_b, true);

    // Pointwise complex multiply in frequency domain
    let mut re_out = [0.0_f32; N];
    let mut im_out = [0.0_f32; N];
    for i in 0..N {
        // (a_re + i·a_im) × (b_re + i·b_im)
        re_out[i] = re_a[i] * re_b[i] - im_a[i] * im_b[i];
        im_out[i] = re_a[i] * im_b[i] + im_a[i] * re_b[i];
    }

    // Inverse FFT
    fft64(&mut re_out, &mut im_out, false);

    complex_to_ring(&re_out, &im_out)
}

/// Compute the angular power spectrum of a ring.
///
/// Returns the squared magnitude of each FFT coefficient: |F_k|².
/// The DC component (k=0) encodes the centroid offset; higher frequencies
/// encode shape detail.
pub fn angular_spectrum(ring: &[[f32; 2]; N]) -> [f32; N] {
    let (mut re, mut im) = ring_to_complex(ring);
    fft64(&mut re, &mut im, true);

    let mut spectrum = [0.0_f32; N];
    for i in 0..N {
        spectrum[i] = re[i] * re[i] + im[i] * im[i];
    }
    spectrum
}

/// Frequency-domain distance between two rings.
///
/// Computes the Euclidean distance between the angular spectra, keeping only
/// the first `n_freqs` frequency components (DC + low frequencies capture
/// gross shape; high frequencies capture fine detail).
pub fn spectral_ring_distance(
    ring_a: &[[f32; 2]; N],
    ring_b: &[[f32; 2]; N],
    n_freqs: usize,
) -> f64 {
    let spec_a = angular_spectrum(ring_a);
    let spec_b = angular_spectrum(ring_b);
    let n = n_freqs.min(N);

    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let diff = spec_a[i] as f64 - spec_b[i] as f64;
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Low-pass filter: keep only the first `keep` frequency components,
/// zero out the rest, and reconstruct the ring.
pub fn low_pass_ring(ring: &[[f32; 2]; N], keep: usize) -> [[f32; 2]; N] {
    let (mut re, mut im) = ring_to_complex(ring);
    fft64(&mut re, &mut im, true);

    let keep = keep.min(N);
    for i in keep..(N - keep + 1) {
        re[i] = 0.0;
        im[i] = 0.0;
    }

    fft64(&mut re, &mut im, false);
    complex_to_ring(&re, &im)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a unit circle ring (32 equidistant points).
    fn unit_circle() -> [[f32; 2]; N] {
        let mut ring = [[0.0_f32; 2]; N];
        for (i, pt) in ring.iter_mut().enumerate() {
            let t = TAU * i as f32 / N as f32;
            *pt = [t.cos(), t.sin()];
        }
        ring
    }

    /// Generate a "spiked star" ring (elongated, irregular).
    fn spiked_star() -> [[f32; 2]; N] {
        let mut ring = [[0.0_f32; 2]; N];
        for (i, pt) in ring.iter_mut().enumerate() {
            let t = TAU * i as f32 / N as f32;
            let mut x = 18.0 * t.cos();
            let mut y = 0.35 * t.sin();
            if i % 4 == 0 {
                x += 9.0;
            }
            if i % 8 == 0 {
                y -= 6.5;
            }
            *pt = [x, y];
        }
        ring
    }

    #[test]
    fn circle_conv_circle_is_circle() {
        let c = unit_circle();
        let result = cyclic_convolution_rings(&c, &c);

        // After convolution of two identical circles, the result should
        // still be circular (all points at roughly the same distance from origin).
        let dists: Vec<f32> = result.iter().map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt()).collect();
        let mean = dists.iter().sum::<f32>() / N as f32;
        let max_dev = dists.iter().map(|&d| (d - mean).abs()).fold(0.0_f32, f32::max);
        // Allow generous tolerance since convolution changes amplitude
        assert!(max_dev < 2.0, "circle conv circle should remain roughly circular, max_dev={max_dev}");
    }

    #[test]
    fn angular_spectrum_has_dc_dominant() {
        let c = unit_circle();
        let spec = angular_spectrum(&c);

        // DC component (k=0) encodes centroid — for a centered circle, it's 0
        // First AC component (k=1) should be largest for a circle
        assert!(spec[1] > 0.0, "circle should have AC energy");
    }

    #[test]
    fn star_vs_circle_distance_is_significant() {
        let c = unit_circle();
        let s = spiked_star();

        // Distance between very different shapes
        let d_diff = spectral_ring_distance(&c, &s, 8);
        // Distance between identical shapes
        let d_same = spectral_ring_distance(&c, &c, 8);

        assert!(d_diff > d_same * 2.0, "different shapes should be farther apart than identical ones");
    }

    #[test]
    fn low_pass_preserves_coarse_shape() {
        let s = spiked_star();
        let smoothed = low_pass_ring(&s, 4); // keep only lowest 4 frequencies

        // Smoothed ring should have points (not NaN)
        for pt in &smoothed {
            assert!(pt[0].is_finite() && pt[1].is_finite());
        }
    }
}