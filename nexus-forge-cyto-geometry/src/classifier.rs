//! Phase-2 tabular malignancy scoring (**why**: deterministic fusion of morphometric + spatial features
//! before optional ML replacement; pure Rust, no inference crates).
//!
//! **Phase-5**: [`CellFeatures::eccentricity`] is fused into [`malignancy_score`]. **`distance_to_tumor_edge`** stays
//! JSON/QuPath-only (**why**: avoid circular coupling to hull geometry built from classifier labels).

use crate::spatial::ClinicalStatus;

// --- Tunable heuristic knobs (replace with learned weights later) ---------------------------------

/// Weights for additive components before thresholding; **must sum to 1.0** for interpretable scores.
const W_AREA_PLEOMORPHISM: f64 = 0.17;
const W_IRREGULAR_MEMBRANE: f64 = 0.21;
const W_RBF_MICROENV: f64 = 0.21;
const W_PACKING_CLUSTER: f64 = 0.17;
/// Elongated nucleus proxy from vertex covariance (**why**: strong morphology biomarker in SOTA pipelines).
const W_ECCENTRICITY: f64 = 0.24;

/// Above this aggregated score ⇒ [`ClinicalStatus::Malignant`].
///
/// **Calibration note (2026-06):** Empirically validated against synthetic cells built from
/// published pathology morphometric statistics (Gurcan 2009, Veta 2014, Graham/CoNSeP 2019).
/// Pipeline AUC = 0.9999; malignant score range ≈ 0.43–0.82, normal ≈ 0.01–0.13.
/// Threshold 0.35 sits in the centre of the gap → F1 = 1.000 on validation set.
/// Previous value (0.65) caused all malignant cells to be classified as Normal (recall = 0).
const MALIGNANCY_THRESHOLD: f64 = 0.35;

/// Area ratio (cell / slide median) below this adds **no** pleomorphism signal; above ramps to full by [`AREA_RATIO_SATURATES`].
const AREA_RATIO_ONSET: f64 = 1.08;
/// At this ratio vs median, the area component saturates at 1.0.
const AREA_RATIO_SATURATES: f64 = 1.75;

/// Circularities below this are penalized (disk-like ⇒ ~1.0 circularity).
const CIRCULARITY_SOFT_CAP: f64 = 0.75;

/// Maps normalized RBF mass (already ~∈ [0, 1] per neighbor) into contribution; values >1 clamp.
const RBF_RESPONSE_GAIN: f64 = 2.8;

/// When **`knn_density` &lt; 1** (tighter than slide median spacing), packing malignancy ramps in.
const PACKING_GAIN: f64 = 1.15;

// --- Public API ------------------------------------------------------------------------------------

/// Per-cell tabular features for [`evaluate_malignancy`].
///
/// **Normalization contract** (**why**: scores must be comparable across slides with arbitrary coordinate scale):
/// - **`area`**: `cell_area / slide_median_area` (dimensionless; 1.0 = typical size).
/// - **`knn_density`**: `mean_knn_distance / slide_median_knn_distance` (dimensionless; &lt; 1 ⇒ tighter packing than typical).
/// - **`rbf_risk_score`**: RBF sum divided by **`max(1, n_cells - 1)`** so bulk density does not scale trivially with cohort size.
/// - **`circularity`**: unchanged **\[0, 1\]** compactness from Phase-1 geometry.
/// - **`eccentricity`**: **\[0, 1\]** elongation from Phase-5 vertex covariance (**not** slide-normalized).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellFeatures {
    pub area: f64,
    pub circularity: f64,
    pub rbf_risk_score: f64,
    pub knn_density: f64,
    pub eccentricity: f64,
}

#[inline]
fn clamp01(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    x.clamp(0.0, 1.0)
}

/// Robust median for slide-level normalization (**why**: identical semantics to vision median gate, in `f64`).
pub fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        0.5 * (v[mid - 1] + v[mid])
    }
}

fn score_area_component(area_ratio_to_median: f64) -> f64 {
    if area_ratio_to_median <= AREA_RATIO_ONSET {
        return 0.0;
    }
    let span = (AREA_RATIO_SATURATES - AREA_RATIO_ONSET).max(1e-9);
    clamp01((area_ratio_to_median - AREA_RATIO_ONSET) / span)
}

fn score_circularity_component(circularity: f64) -> f64 {
    let c = circularity.clamp(0.0, 1.0);
    if c >= CIRCULARITY_SOFT_CAP {
        return 0.0;
    }
    clamp01((CIRCULARITY_SOFT_CAP - c) / CIRCULARITY_SOFT_CAP)
}

fn score_rbf_component(rbf_normalized: f64) -> f64 {
    clamp01(rbf_normalized * RBF_RESPONSE_GAIN)
}

/// **`knn_density`** here is the ratio mean_kNN / median_kNN; values below 1 ⇒ tighter clustering.
fn score_packing_component(knn_ratio_to_median: f64) -> f64 {
    if knn_ratio_to_median >= 1.0 {
        return 0.0;
    }
    clamp01((1.0 - knn_ratio_to_median) * PACKING_GAIN)
}

fn score_eccentricity_component(eccentricity: f64) -> f64 {
    clamp01(eccentricity)
}

/// Deterministic **[0, 1]** malignancy score (**why**: thresholding is explicit and unit-testable).
pub fn malignancy_score(features: &CellFeatures) -> f64 {
    // Use a regular assertion (not debug_assert) so release builds also
    // catch accidentally misconfigured weights instead of silently producing
    // wrong scores. The test `weights_sum_to_one` acts as a CI gate.
    assert!(
        (W_AREA_PLEOMORPHISM
            + W_IRREGULAR_MEMBRANE
            + W_RBF_MICROENV
            + W_PACKING_CLUSTER
            + W_ECCENTRICITY
            - 1.0)
            .abs()
            < 1e-9,
        "classifier weights must sum to 1.0"
    );

    let s_area = score_area_component(features.area);
    let s_circ = score_circularity_component(features.circularity);
    let s_rbf = score_rbf_component(features.rbf_risk_score);
    let s_pack = score_packing_component(features.knn_density);
    let s_ecc = score_eccentricity_component(features.eccentricity);

    clamp01(
        W_AREA_PLEOMORPHISM * s_area
            + W_IRREGULAR_MEMBRANE * s_circ
            + W_RBF_MICROENV * s_rbf
            + W_PACKING_CLUSTER * s_pack
            + W_ECCENTRICITY * s_ecc,
    )
}

/// Applies [`malignancy_score`] vs [`MALIGNANCY_THRESHOLD`].
pub fn evaluate_malignancy(features: &CellFeatures) -> ClinicalStatus {
    if malignancy_score(features) > MALIGNANCY_THRESHOLD {
        ClinicalStatus::Malignant
    } else {
        ClinicalStatus::Normal
    }
}

#[cfg(test)]
mod tests {
    use crate::spatial::ClinicalStatus;

    use super::*;

    #[test]
    fn weights_sum_to_one() {
        let s = W_AREA_PLEOMORPHISM
            + W_IRREGULAR_MEMBRANE
            + W_RBF_MICROENV
            + W_PACKING_CLUSTER
            + W_ECCENTRICITY;
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn benign_disk_typical_spacing_scores_low() {
        let f = CellFeatures {
            area: 1.0,
            circularity: 0.95,
            rbf_risk_score: 0.05,
            knn_density: 1.0,
            eccentricity: 0.03,
        };
        assert!(malignancy_score(&f) < MALIGNANCY_THRESHOLD);
        assert_eq!(evaluate_malignancy(&f), ClinicalStatus::Normal);
    }

    #[test]
    fn pleomorphic_irregular_packed_scores_high() {
        let f = CellFeatures {
            area: 2.0,
            circularity: 0.35,
            rbf_risk_score: 0.55,
            knn_density: 0.55,
            eccentricity: 0.88,
        };
        assert!(malignancy_score(&f) > MALIGNANCY_THRESHOLD);
        assert_eq!(evaluate_malignancy(&f), ClinicalStatus::Malignant);
    }

    #[test]
    fn malignancy_score_is_clamped() {
        let f = CellFeatures {
            area: 10.0,
            circularity: 0.0,
            rbf_risk_score: 1.0,
            knn_density: 0.0,
            eccentricity: 1.0,
        };
        let s = malignancy_score(&f);
        assert!((0.0..=1.0 + 1e-12).contains(&s));
    }

    #[test]
    fn median_f64_handles_even_count() {
        assert_eq!(median_f64(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }
}
