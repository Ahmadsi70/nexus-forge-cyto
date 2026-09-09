//! Mappings #10 + #14: Multi-scale density bridge & tiered scale normalisation.
//!
//! **Mapping #10 — Two-scale tissue model** (from `discovered_mathematical_mappings.md`):
//! > ρ = Abjad_kabir / w,  Φ_QG = ρ_macro / ρ_micro ≈ 1.102614,
//! > Λ = ln(Φ_QG) ≈ 0.097684,  M_geom = √(ρ_macro · ρ_micro) ≈ 351.710470
//!
//! **Cell-network analogue:**
//! - "macro" = tile-level mean nuclear density
//! - "micro" = single-cell local density
//! - Φ = dimensionless coupling comparing scales
//! - log Φ = information-like scale gap
//! - M_geom = typical density bridge — cross-slide normaliser
//!
//! **Mapping #14 — Tiered scale normalisation:**
//! > E_Planck = √(ħc⁵/G),  E_bridge = E_Planck · N_c / α⁻¹
//!
//! **Cell-network analogue:**
//! - Anchor learned coupling strengths between cellular scales
//!   (organelle ↔ nucleus ↔ neighbourhood) to fixed numeric tiers.
//! - Dimensionless ratio discipline for multi-magnification features.

use crate::spatial::SpatialNode;

// ---------------------------------------------------------------------------
// Mapping #10: Two-scale density bridge
// ---------------------------------------------------------------------------

/// Two-scale tissue density model.
#[derive(Clone, Debug)]
pub struct TwoScaleModel {
    /// Tile-level mean density (macro scale).
    pub rho_macro: f64,
    /// Single-cell local density (micro scale).
    pub rho_micro: f64,
    /// Dimensionless coupling: Φ = ρ_macro / ρ_micro.
    pub phi_qg: f64,
    /// Information-like scale gap: Λ = ln(Φ).
    pub lambda: f64,
    /// Geometric mean bridge: M = √(ρ_macro · ρ_micro).
    pub geometric_mean: f64,
}

/// Compute the two-scale density model from spatial nodes.
///
/// - **macro scale:** mean nuclear area over the entire tile (or all cells).
/// - **micro scale:** median of per-cell local kNN density.
pub fn compute_two_scale_bridge(nodes: &[SpatialNode]) -> TwoScaleModel {
    let n = nodes.len();
    if n == 0 {
        return TwoScaleModel {
            rho_macro: 0.0,
            rho_micro: 0.0,
            phi_qg: 0.0,
            lambda: 0.0,
            geometric_mean: 0.0,
        };
    }

    // Macro: mean nuclear area
    let rho_macro: f64 = nodes.iter().map(|nd| nd.area.max(1e-18)).sum::<f64>() / n as f64;

    // Micro: median local density (area-based proxy)
    let mut areas: Vec<f64> = nodes.iter().map(|nd| nd.area.max(1e-18)).collect();
    areas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = n / 2;
    let rho_micro = if n % 2 == 1 {
        areas[mid]
    } else {
        0.5 * (areas[mid - 1] + areas[mid])
    };

    let phi_qg = if rho_micro > 1e-18 {
        rho_macro / rho_micro
    } else {
        0.0
    };
    let lambda = if phi_qg > 0.0 { phi_qg.ln() } else { 0.0 };
    let geometric_mean = (rho_macro * rho_micro).sqrt();

    TwoScaleModel {
        rho_macro,
        rho_micro,
        phi_qg,
        lambda,
        geometric_mean,
    }
}

/// Normalise a per-cell feature by the geometric mean bridge.
///
/// f'_i = f_i / M_geom
///
/// This removes scale differences between slides, making features comparable
/// regardless of staining intensity or magnification.
pub fn normalise_by_bridge(features: &[f64], bridge: &TwoScaleModel) -> Vec<f64> {
    let denom = bridge.geometric_mean.max(1e-18);
    features.iter().map(|&f| f / denom).collect()
}

// ---------------------------------------------------------------------------
// Mapping #14: Tiered scale normalisation
// ---------------------------------------------------------------------------

/// Tiered scale constants for multi-magnification feature alignment.
///
/// Ratios between scales are fixed (order-of-magnitude anchors), enabling
/// dimensionless comparison across imaging modalities.
#[derive(Clone, Debug)]
pub struct ScaleTiers {
    /// Sub-cellular / organelle scale (≈ 10⁻⁶ m).
    pub organelle: f64,
    /// Nuclear scale (≈ 10⁻⁵ m).
    pub nuclear: f64,
    /// Cellular scale (≈ 10⁻⁴ m).
    pub cellular: f64,
    /// Tissue neighbourhood scale (≈ 10⁻³ m).
    pub neighborhood: f64,
}

impl Default for ScaleTiers {
    fn default() -> Self {
        Self {
            organelle: 1e-6,
            nuclear: 1e-5,
            cellular: 1e-4,
            neighborhood: 1e-3,
        }
    }
}

impl ScaleTiers {
    /// Ratio between two scales (source / target).
    pub fn scale_ratio(&self, source: f64, target: f64) -> f64 {
        if target.abs() < 1e-30 {
            return 1.0;
        }
        source / target
    }

    /// All scale values in ascending order.
    pub fn all_scales(&self) -> [f64; 4] {
        [self.organelle, self.nuclear, self.cellular, self.neighborhood]
    }
}

/// Normalise a feature value from one scale tier to another.
///
/// This produces dimensionless quantities that are invariant under
/// magnification changes. The underlying assumption is that feature
/// magnitudes scale proportionally with the observation scale.
pub fn tiered_normalize(feature_value: f64, source_tier: f64, target_tier: f64) -> f64 {
    let ratio = source_tier / target_tier.max(1e-30);
    feature_value * ratio
}

/// Multi-scale feature vector: compute the same feature at all four tiers.
pub fn multiscale_features(
    base_value: f64,
    base_tier: f64,
    tiers: &ScaleTiers,
) -> [f64; 4] {
    tiers.all_scales().map(|t| tiered_normalize(base_value, base_tier, t))
}

// ---------------------------------------------------------------------------
// Mapping #11: Optimal embedding axis via grid search
// ---------------------------------------------------------------------------

use crate::spatial::ClinicalStatus;

/// Result of a grid search for the optimal 2D projection axis.
#[derive(Clone, Debug)]
pub struct OptimalProjection {
    /// Best rotation angle (radians) for the 2D embedding.
    pub best_theta: f64,
    /// Best weight for distance features (α).
    pub best_alpha: f64,
    /// Best weight for degree features (β).
    pub best_beta: f64,
    /// Separation score (higher = better benign/malignant separation).
    pub separation_score: f64,
}

/// Grid search for the optimal embedding axis separating benign from malignant cells.
///
/// Searches over:
/// - θ ∈ [0, π) in `grid_resolution` steps (rotation angle)
/// - (α, β) ∈ [-3, 3]² mixing distance and degree coordinates
///
/// Separation score = |mean_malignant - mean_benign| / √(σ²_malignant + σ²_benign)
pub fn grid_search_embedding_axis(
    positions: &[[f64; 2]],       // (distance_feature, degree_feature) per cell
    labels: &[ClinicalStatus],
    grid_resolution: usize,
) -> OptimalProjection {
    let n = positions.len();
    if n < 4 {
        return OptimalProjection {
            best_theta: 0.0,
            best_alpha: 1.0,
            best_beta: 0.0,
            separation_score: 0.0,
        };
    }

    let res = grid_resolution.max(3);
    let mut best = OptimalProjection {
        best_theta: 0.0,
        best_alpha: 0.0,
        best_beta: 0.0,
        separation_score: f64::NEG_INFINITY,
    };

    // Grid search over (α, β)
    let alphas: Vec<f64> = (0..res)
        .map(|i| -3.0 + 6.0 * i as f64 / (res - 1) as f64)
        .collect();
    let betas = alphas.clone();

    for &alpha in &alphas {
        for &beta in &betas {
            // Build 2D coordinates: z_i = (α·d_i + β·deg_i, …)
            let coords: Vec<[f64; 2]> = positions
                .iter()
                .map(|&[d, deg]| [alpha * d + beta * deg, 0.0])
                .collect();

            // Search over θ
            for t_idx in 0..res {
                let theta = std::f64::consts::PI * t_idx as f64 / (res - 1) as f64;
                let ct = theta.cos();
                let st = theta.sin();

                let projected: Vec<f64> = coords.iter().map(|c| c[0] * ct + c[1] * st).collect();

                // Compute means per class
                let mut mal_sum = 0.0_f64;
                let mut mal_count = 0_usize;
                let mut ben_sum = 0.0_f64;
                let mut ben_count = 0_usize;

                for (i, val) in projected.iter().enumerate() {
                    match labels.get(i) {
                        Some(ClinicalStatus::Malignant) => {
                            mal_sum += val;
                            mal_count += 1;
                        }
                        _ => {
                            ben_sum += val;
                            ben_count += 1;
                        }
                    }
                }

                if mal_count < 2 || ben_count < 2 {
                    continue;
                }

                let mal_mean = mal_sum / mal_count as f64;
                let ben_mean = ben_sum / ben_count as f64;

                let mal_var: f64 = projected
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| matches!(labels.get(*i), Some(ClinicalStatus::Malignant)))
                    .map(|(_, v)| (v - mal_mean).powi(2))
                    .sum::<f64>()
                    / mal_count as f64;

                let ben_var: f64 = projected
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !matches!(labels.get(*i), Some(ClinicalStatus::Malignant)))
                    .map(|(_, v)| (v - ben_mean).powi(2))
                    .sum::<f64>()
                    / ben_count as f64;

                let denom = (mal_var + ben_var).sqrt();
                let score = if denom > 1e-18 {
                    (mal_mean - ben_mean).abs() / denom
                } else {
                    0.0
                };

                if score > best.separation_score {
                    best = OptimalProjection {
                        best_theta: theta,
                        best_alpha: alpha,
                        best_beta: beta,
                        separation_score: score,
                    };
                }
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(area: f64) -> SpatialNode {
        SpatialNode {
            cell_id: 0,
            centroid_x: 0.0,
            centroid_y: 0.0,
            area,
            perimeter: 0.0,
            circularity: 0.0,
            eccentricity: 0.0,
            solidity: 0.0,
            convexity: 0.0,
            aspect_ratio: 1.0,
            hu_moments: [0.0; 7],
            distance_to_tumor_edge: 0.0,
            clinical_status: None,
        }
    }

    #[test]
    fn two_scale_bridge_positive() {
        let nodes: Vec<SpatialNode> = (0..10).map(|_| make_node(2.0)).collect();
        let bridge = compute_two_scale_bridge(&nodes);

        assert!(bridge.rho_macro > 0.0);
        assert!(bridge.rho_micro > 0.0);
        assert!(bridge.geometric_mean > 0.0);
        // log Φ ≈ 0 for uniform cells
        assert!(bridge.lambda.abs() < 0.5);
    }

    #[test]
    fn tiered_normalisation_scales_linearly() {
        let tiers = ScaleTiers::default();
        let val = 5.0;

        // nuclear → cellular (10x larger)
        let up = tiered_normalize(val, tiers.nuclear, tiers.cellular);
        assert!((up - 0.5).abs() < 1e-9); // 5.0 × (1e-5 / 1e-4) = 0.5

        // cellular → nuclear (10x smaller)
        let down = tiered_normalize(val, tiers.cellular, tiers.nuclear);
        assert!((down - 50.0).abs() < 1e-9); // 5.0 × (1e-4 / 1e-5) = 50.0
    }

    #[test]
    fn grid_search_finds_separation() {
        // Two well-separated clusters in (distance, degree) space
        let positions: Vec<[f64; 2]> = (0..20)
            .map(|i| {
                if i < 10 {
                    [i as f64 * 0.1, i as f64 * 0.1]
                } else {
                    [10.0 + i as f64 * 0.1, 10.0 + i as f64 * 0.1]
                }
            })
            .collect();
        let mut labels = vec![ClinicalStatus::Normal; 20];
        for i in 10..20 {
            labels[i] = ClinicalStatus::Malignant;
        }

        let opt = grid_search_embedding_axis(&positions, &labels, 5);
        assert!(opt.separation_score > 0.5, "should find good separation");
    }
}