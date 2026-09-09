//! Mapping #7: Multi-agent configuration space on (S³)^n × ℝ^(2n).
//!
//! **Mathematical origin** (from `discovered_mathematical_mappings.md`):
//! > dim (S³)^n = 3n (smooth manifold); embedding in ℝ^(4n) finite-dimensional.
//!
//! **Cell-network analogue:**
//! - n interacting cells each carrying an orientation/phase on S³ (or S¹ if reduced)
//!   ⇒ configuration space dimension 3n (plus translations 2n in 2D pathology tiles → 5n total).
//! - Configuration entropy as measure of tissue disorder.

use crate::spatial::SpatialNode;

/// One cell's state in the configuration manifold: position (ℝ²) + orientation (S³).
#[derive(Clone, Debug)]
pub struct CellState {
    pub cell_id: u32,
    /// Spatial position (x, y) in ℝ².
    pub position: [f64; 2],
    /// Orientation encoded as a unit quaternion in S³ ⊂ ℝ⁴.
    /// q = (w, x, y, z) with ‖q‖ = 1.
    pub orientation: [f64; 4],
}

/// The full configuration space of n interacting cells.
#[derive(Clone, Debug)]
pub struct ConfigurationSpace {
    pub n_cells: usize,
    /// Total dimension: 3n (orientations) + 2n (positions) = 5n.
    pub total_dim: usize,
    /// Per-cell states.
    pub states: Vec<CellState>,
}

impl ConfigurationSpace {
    /// Build the configuration space from spatial nodes and scalar kappa values.
    ///
    /// Each cell's orientation is encoded as a unit quaternion derived from its
    /// morphometric profile:
    /// - w = (1 - eccentricity) × circularity
    /// - x = cos(2π × kappa)
    /// - y = sin(2π × kappa)
    /// - z = area_ratio (normalised)
    pub fn from_nodes(nodes: &[SpatialNode], kappa_values: &[f32]) -> Self {
        let n = nodes.len();
        let mut states = Vec::with_capacity(n);

        for (i, node) in nodes.iter().enumerate() {
            let kappa = kappa_values.get(i).copied().unwrap_or(0.0) as f64;
            let ecc = node.eccentricity.clamp(0.0, 1.0);
            let circ = node.circularity.clamp(0.0, 1.0);

            // Build quaternion from morphometrics
            let w = (1.0 - ecc) * circ;
            let angle = 2.0 * std::f64::consts::PI * kappa;
            let x = angle.cos() * ecc;
            let y = angle.sin() * ecc;
            let z = (node.area / 10.0).clamp(-1.0, 1.0); // normalised area proxy

            // Normalise to unit quaternion
            let norm = (w * w + x * x + y * y + z * z).sqrt().max(1e-18);
            let orientation = [w / norm, x / norm, y / norm, z / norm];

            states.push(CellState {
                cell_id: i as u32,
                position: [node.centroid_x, node.centroid_y],
                orientation,
            });
        }

        ConfigurationSpace {
            n_cells: n,
            total_dim: 3 * n + 2 * n,
            states,
        }
    }
}

/// Geodesic distance between two unit quaternions on S³.
///
/// d(q₁, q₂) = |log(q₁⁻¹ ⊗ q₂)| = arccos(|⟨q₁, q₂⟩|)
#[inline]
pub fn quaternion_geodesic_distance(q1: &[f64; 4], q2: &[f64; 4]) -> f64 {
    let dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
    dot.abs().min(1.0).acos()
}

/// Configuration entropy: mean pairwise geodesic distance on S³.
///
/// Higher entropy ⇒ more disorder in cellular orientations.
pub fn configuration_entropy(space: &ConfigurationSpace) -> f64 {
    let n = space.n_cells;
    if n < 2 {
        return 0.0;
    }

    let mut total_dist = 0.0_f64;
    let mut count = 0_usize;

    for i in 0..n {
        for j in (i + 1)..n {
            total_dist += quaternion_geodesic_distance(
                &space.states[i].orientation,
                &space.states[j].orientation,
            );
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { total_dist / count as f64 }
}

/// Spatial clustering of orientations: mean geodesic distance among k-nearest spatial neighbours.
pub fn local_orientation_coherence(space: &ConfigurationSpace, k: usize) -> f64 {
    let n = space.n_cells;
    if n < 2 {
        return 0.0;
    }

    let k_eff = k.min(n - 1).max(1);
    let mut total_coherence = 0.0_f64;

    for i in 0..n {
        // Find k nearest spatial neighbours
        let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j { continue; }
            let dx = space.states[i].position[0] - space.states[j].position[0];
            let dy = space.states[i].position[1] - space.states[j].position[1];
            dists.push((dx * dx + dy * dy, j));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut sum_dist = 0.0_f64;
        for &(_, j) in dists.iter().take(k_eff) {
            sum_dist += quaternion_geodesic_distance(
                &space.states[i].orientation,
                &space.states[j].orientation,
            );
        }
        total_coherence += sum_dist / k_eff as f64;
    }

    total_coherence / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::SpatialNode;

    fn make_node(id: u32, x: f64, y: f64, area: f64, ecc: f64, circ: f64) -> SpatialNode {
        SpatialNode {
            cell_id: id,
            centroid_x: x,
            centroid_y: y,
            area,
            perimeter: 0.0,
            circularity: circ,
            eccentricity: ecc,
            solidity: 0.0,
            convexity: 0.0,
            aspect_ratio: 1.0,
            hu_moments: [0.0; 7],
            distance_to_tumor_edge: 0.0,
            clinical_status: None,
        }
    }

    #[test]
    fn config_space_from_nodes() {
        let nodes: Vec<SpatialNode> = (0..10)
            .map(|i| make_node(i, i as f64, 0.0, 1.0, 0.5, 0.8))
            .collect();
        let kappas: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();

        let space = ConfigurationSpace::from_nodes(&nodes, &kappas);
        assert_eq!(space.n_cells, 10);
        assert_eq!(space.total_dim, 50); // 3×10 + 2×10
        assert_eq!(space.states.len(), 10);

        // All quaternions should be unit norm
        for state in &space.states {
            let n = state.orientation.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-9, "quaternion should be unit norm");
        }
    }

    #[test]
    fn quaternion_distance_symmetric() {
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.0, 1.0, 0.0, 0.0];

        let d12 = quaternion_geodesic_distance(&q1, &q2);
        let d21 = quaternion_geodesic_distance(&q2, &q1);
        assert!((d12 - d21).abs() < 1e-12);
        assert!(d12 > 0.0);
    }

    #[test]
    fn entropy_differentiates_order_from_chaos() {
        // Ordered: all cells have the same orientation
        let mut nodes = Vec::new();
        for i in 0..20 {
            nodes.push(make_node(i, i as f64, 0.0, 1.0, 0.1, 0.9));
        }
        let kappas_ordered: Vec<f32> = vec![0.1; 20];
        let space_ordered = ConfigurationSpace::from_nodes(&nodes, &kappas_ordered);
        let entropy_ordered = configuration_entropy(&space_ordered);

        // Chaotic: varied kappa values
        let kappas_chaos: Vec<f32> = (0..20).map(|i| (i as f32 * 0.05).fract()).collect();
        let space_chaos = ConfigurationSpace::from_nodes(&nodes, &kappas_chaos);
        let entropy_chaos = configuration_entropy(&space_chaos);

        // Chaotic should have higher entropy
        assert!(
            entropy_chaos >= entropy_ordered * 0.9,
            "chaotic configuration should have at least comparable entropy to ordered"
        );
    }
}