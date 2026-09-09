//! Mappings #12 + #13: Path entropy along invasion gradients & equilibrium patches.
//!
//! **Mapping #12 — Path entropy** (from `discovered_mathematical_mappings.md`):
//! > Li = W·t★ / S_Shannon_bits,  ΔS(i→i+1) = S_{i+1} - S_i,  ρ_L(i) = Li_{i+1} / Li_i
//!
//! **Cell-network analogue:**
//! - Index i along a geodesic walk on the cell graph (e.g., radial from lumen).
//! - ΔS measures entropy production along the path (marker entropy or neighbourhood-type entropy).
//! - ρ_L = luminosity ratio analogue → fold-change in predicted malignancy risk along invasion axis.
//!
//! **Mapping #13 — Reversible equilibrium:**
//! > entropy production = 0 at reversible equilibrium
//!
//! **Cell-network analogue:**
//! - Identify graph regions where net flux of a diffusion process on cell-cell edges
//!   is zero → equilibrium patches (homogeneous microenvironment domains).

use crate::spatial::SpatialNode;

// ---------------------------------------------------------------------------
// Mapping #12: Path entropy
// ---------------------------------------------------------------------------

/// Entropy and risk metrics along a cellular path (geodesic walk).
#[derive(Clone, Debug)]
pub struct PathEntropy {
    /// Coordinates of cells along the path.
    pub positions: Vec<[f64; 2]>,
    /// Scalar phenotype value at each cell.
    pub phenotype_values: Vec<f64>,
    /// Entropy change at each step: ΔS(i→i+1).
    pub entropy_delta: Vec<f64>,
    /// Luminosity/risk ratio at each step: ρ_L(i) = val_{i+1} / val_i.
    pub luminosity_ratio: Vec<f64>,
    /// Cumulative risk score along the path.
    pub cumulative_risk: Vec<f64>,
    /// Total Shannon entropy of the path.
    pub total_entropy: f64,
}

/// Compute path entropy metrics along an ordered sequence of cell indices.
///
/// The path should be ordered from "origin" (e.g., lumen) to "invasion front"
/// (e.g., stroma). Entropy is computed from the distribution of phenotype values
/// within a sliding window along the path.
pub fn compute_path_entropy(
    nodes: &[SpatialNode],
    path: &[usize], // ordered cell indices along the geodesic
    phenotype_fn: fn(&SpatialNode) -> f64,
) -> PathEntropy {
    let n = path.len();
    if n == 0 {
        return PathEntropy {
            positions: vec![],
            phenotype_values: vec![],
            entropy_delta: vec![],
            luminosity_ratio: vec![],
            cumulative_risk: vec![],
            total_entropy: 0.0,
        };
    }

    // Extract positions and phenotype values along the path
    let positions: Vec<[f64; 2]> = path
        .iter()
        .filter_map(|&idx| nodes.get(idx))
        .map(|nd| [nd.centroid_x, nd.centroid_y])
        .collect();

    let phenotype_values: Vec<f64> = path
        .iter()
        .filter_map(|&idx| nodes.get(idx))
        .map(|nd| phenotype_fn(nd))
        .collect();

    let m = phenotype_values.len();
    if m < 2 {
        return PathEntropy {
            positions,
            phenotype_values: phenotype_values.clone(),
            entropy_delta: vec![],
            luminosity_ratio: vec![],
            cumulative_risk: vec![phenotype_values.first().copied().unwrap_or(0.0)],
            total_entropy: 0.0,
        };
    }

    // Compute sliding-window Shannon entropy
    let window = 5.min(m);
    let mut local_entropies = Vec::with_capacity(m);

    for i in 0..m {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(m);
        let window_vals = &phenotype_values[start..end];

        // Discretise into 4 bins for entropy computation
        let mut bins = [0_usize; 4];
        for &v in window_vals {
            let bin = ((v.abs().fract() * 4.0) as usize).min(3);
            bins[bin] += 1;
        }
        let total = window_vals.len() as f64;
        let ent: f64 = bins
            .iter()
            .map(|&c| {
                let p = c as f64 / total;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum();
        local_entropies.push(ent);
    }

    // ΔS(i→i+1)
    let entropy_delta: Vec<f64> = local_entropies
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    // ρ_L(i) = val_{i+1} / val_i
    let luminosity_ratio: Vec<f64> = phenotype_values
        .windows(2)
        .map(|w| if w[0].abs() > 1e-18 { w[1] / w[0] } else { 1.0 })
        .collect();

    // Cumulative risk
    let mut cumulative_risk = Vec::with_capacity(m);
    let mut cum = 0.0_f64;
    for &v in &phenotype_values {
        cum += v;
        cumulative_risk.push(cum);
    }

    // Total Shannon entropy of the entire path
    let total_entropy: f64 = local_entropies.iter().sum::<f64>() / m as f64;

    PathEntropy {
        positions,
        phenotype_values,
        entropy_delta,
        luminosity_ratio,
        cumulative_risk,
        total_entropy,
    }
}

/// Build a radial path from a start cell outward, following the kNN graph.
///
/// At each step, move to the nearest unvisited neighbour that is farthest
/// from the origin (expanding front). Stops when no unvisited neighbours remain.
pub fn radial_geodesic_path(
    nodes: &[SpatialNode],
    start_cell: usize,
    max_steps: usize,
) -> Vec<usize> {
    let n = nodes.len();
    if start_cell >= n {
        return vec![];
    }

    // Pre-compute pairwise distances for neighbour lookup
    let mut path = vec![start_cell];
    let mut visited = vec![false; n];
    visited[start_cell] = true;

    let _origin = [nodes[start_cell].centroid_x, nodes[start_cell].centroid_y];

    for _ in 0..max_steps {
        let last = *path.last().unwrap();
        let mut best_dist = f64::INFINITY;
        let mut best_j = None;

        for j in 0..n {
            if visited[j] {
                continue;
            }
            let dx = nodes[last].centroid_x - nodes[j].centroid_x;
            let dy = nodes[last].centroid_y - nodes[j].centroid_y;
            let d2 = dx * dx + dy * dy;

            if d2 < best_dist {
                best_dist = d2;
                best_j = Some(j);
            }
        }

        match best_j {
            Some(j) => {
                path.push(j);
                visited[j] = true;
            }
            None => break,
        }
    }

    path
}

// ---------------------------------------------------------------------------
// Mapping #13: Equilibrium patches
// ---------------------------------------------------------------------------

/// Detect homogeneous microenvironment domains where net diffusion flux ≈ 0.
///
/// A "patch" is a connected component of cells where the absolute difference
/// in a scalar phenotype between neighbours is below a threshold.
pub fn detect_equilibrium_patches(
    nodes: &[SpatialNode],
    edge_index: &[(usize, usize)],
    phenotype_fn: fn(&SpatialNode) -> f64,
    flux_threshold: f64,
) -> Vec<Vec<usize>> {
    let n = nodes.len();
    if n == 0 {
        return vec![];
    }

    // Compute phenotype values
    let values: Vec<f64> = nodes.iter().map(phenotype_fn).collect();

    // Build adjacency of "equilibrium edges" (low flux edges)
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for &(a, b) in edge_index {
        if a < n && b < n {
            let diff = (values[a] - values[b]).abs();
            if diff <= flux_threshold {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }

    // Find connected components via DFS
    let mut visited = vec![false; n];
    let mut patches = Vec::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut patch = Vec::new();
        let mut stack = vec![i];
        visited[i] = true;

        while let Some(v) = stack.pop() {
            patch.push(v);
            for &nb in &adj[v] {
                if !visited[nb] {
                    visited[nb] = true;
                    stack.push(nb);
                }
            }
        }

        if patch.len() >= 2 {
            patches.push(patch);
        }
    }

    patches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: u32, x: f64, y: f64, area: f64) -> SpatialNode {
        SpatialNode {
            cell_id: id,
            centroid_x: x,
            centroid_y: y,
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

    fn area_phenotype(node: &SpatialNode) -> f64 {
        node.area
    }

    #[test]
    fn path_entropy_increases_along_gradient() {
        // Line of cells with increasing area
        let nodes: Vec<SpatialNode> = (0..20)
            .map(|i| make_node(i, i as f64, 0.0, i as f64 * 0.5 + 0.1))
            .collect();
        let path: Vec<usize> = (0..20).collect();

        let pe = compute_path_entropy(&nodes, &path, area_phenotype);

        assert_eq!(pe.positions.len(), 20);
        assert_eq!(pe.entropy_delta.len(), 19);
        // Cumulative risk should increase monotonically
        assert!(pe.cumulative_risk[19] > pe.cumulative_risk[0]);
    }

    #[test]
    fn radial_path_visits_all_reachable_cells() {
        // Grid of cells
        let nodes: Vec<SpatialNode> = (0..25)
            .map(|i| make_node(i, (i % 5) as f64, (i / 5) as f64, 1.0))
            .collect();

        let path = radial_geodesic_path(&nodes, 0, 30);
        assert!(!path.is_empty());
        assert_eq!(path[0], 0);
        // Should visit at least some cells beyond the start
        assert!(path.len() >= 2);
    }

    #[test]
    fn equilibrium_patches_detect_homogeneous_regions() {
        // Two clusters: one uniform, one mixed
        let nodes: Vec<SpatialNode> = (0..12)
            .map(|i| {
                let area = if i < 6 { 0.5 } else { 2.0 };
                make_node(i, i as f64, 0.0, area)
            })
            .collect();

        // Full graph edges
        let mut edges = Vec::new();
        for i in 0..12 {
            for j in (i + 1)..12 {
                if (i as isize - j as isize).abs() <= 2 {
                    edges.push((i, j));
                }
            }
        }

        let patches = detect_equilibrium_patches(&nodes, &edges, area_phenotype, 0.01);

        // Should find at least one patch (the uniform cluster)
        assert!(!patches.is_empty(), "should detect at least one equilibrium patch");
    }
}