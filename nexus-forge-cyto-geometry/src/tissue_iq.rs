//! Mapping #3: Module-wise tissue IQ (heterogeneity index).
//!
//! **Mathematical origin** (from `discovered_mathematical_mappings.md`):
//! > Iq(module) = unique_root_count / mean(abjad_density) over verses in module
//! > mean_iq_across_modules ≈ 0.504, CV ≈ 83.47%
//!
//! **Cell-network analogue:**
//! - Partition slide into spatial modules (graph communities via Louvain-like aggregation).
//! - For each module m: IQ_tissue(m) = #distinct_phenotypes / mean_nuclear_density.
//! - High variance across modules → heterogeneity index (analogous to cross-module CV).
//!
//! **Algorithm:**
//! 1. Build kNN graph on cell centroids.
//! 2. Greedy community detection (modularity-inspired merging).
//! 3. Compute IQ per module.
//! 4. Compute cross-module statistics and identify heterogeneous regions.

use crate::spatial::{build_knn_graph, SpatialNode, DEFAULT_KNN_K};

/// Tissue IQ result for a single spatial module.
#[derive(Clone, Debug)]
pub struct ModuleIQ {
    /// Indices of cells belonging to this module.
    pub cell_indices: Vec<usize>,
    /// Number of unique phenotypes in this module.
    pub unique_phenotypes: usize,
    /// Mean nuclear feature density (proxy: mean area).
    pub mean_nuclear_density: f64,
    /// IQ score: unique_phenotypes / mean_nuclear_density.
    pub iq_score: f64,
}

/// Global tissue IQ analysis across all modules.
#[derive(Clone, Debug)]
pub struct TissueIQ {
    /// Per-module IQ results.
    pub modules: Vec<ModuleIQ>,
    /// Mean IQ across all modules.
    pub mean_iq: f64,
    /// Population standard deviation of IQ scores.
    pub std_iq: f64,
    /// Coefficient of variation (CV) of IQ scores.
    pub cv_iq: f64,
    /// Indices of modules with IQ > mean + 1 std (highly heterogeneous).
    pub heterogeneous_regions: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Community detection: greedy Louvain-inspired modularity merging
// ---------------------------------------------------------------------------

/// Greedy community detection on the kNN graph.
///
/// Starts with each cell as its own community, then iteratively merges
/// communities that share the most edges until the target number of modules
/// is reached or no more beneficial merges exist.
///
/// **Why greedy instead of full Louvain**: Deterministic, no random seed,
/// and sufficient for module-wise IQ on slides with ≤1000 cells.
pub fn detect_communities(
    n_cells: usize,
    knn_edges: &[(usize, usize)],
    target_modules: usize,
) -> Vec<Vec<usize>> {
    if n_cells == 0 {
        return vec![];
    }
    if n_cells <= target_modules {
        return (0..n_cells).map(|i| vec![i]).collect();
    }

    // Initialise: each cell is its own community
    let mut community_of: Vec<usize> = (0..n_cells).collect();
    let mut communities: Vec<Vec<usize>> = (0..n_cells).map(|i| vec![i]).collect();

    // Count inter-community edge weights
    let mut inter_edges: Vec<Vec<usize>> = vec![vec![0; n_cells]; n_cells];
    for &(a, b) in knn_edges {
        if a < n_cells && b < n_cells {
            inter_edges[a][b] += 1;
            inter_edges[b][a] += 1;
        }
    }

    while communities.len() > target_modules {
        // Find the pair of communities with the most inter-connections
        let mut best_count = 0;
        let mut best_pair = (0, 0);

        for i in 0..communities.len() {
            for j in (i + 1)..communities.len() {
                let mut count = 0;
                for &ca in &communities[i] {
                    for &cb in &communities[j] {
                        count += inter_edges[ca][cb];
                    }
                }
                if count > best_count {
                    best_count = count;
                    best_pair = (i, j);
                }
            }
        }

        if best_count == 0 {
            break; // no more inter-community edges
        }

        // Merge community j into i
        let (i, j) = best_pair;
        let merged_cells = communities[j].clone();
        for &cell in &merged_cells {
            community_of[cell] = i;
        }
        communities[i].extend(merged_cells);
        communities.remove(j);
    }

    communities
}

// ---------------------------------------------------------------------------
// IQ computation
// ---------------------------------------------------------------------------

/// Compute tissue IQ for each module.
///
/// IQ(module) = #distinct_phenotypes / mean_nuclear_density
///
/// where `phenotype_fn` maps a cell to a discrete phenotype label (0, 1, 2, …),
/// and nuclear density is computed from the cell's area (proxy for nuclear size).
pub fn compute_tissue_iq(
    nodes: &[SpatialNode],
    modules: &[Vec<usize>],
    phenotype_fn: fn(&SpatialNode) -> u8,
) -> TissueIQ {
    let mut module_iqs = Vec::with_capacity(modules.len());

    for module in modules {
        if module.is_empty() {
            continue;
        }

        // Count unique phenotypes
        let mut phenotypes: Vec<u8> = module.iter().map(|&i| phenotype_fn(&nodes[i])).collect();
        phenotypes.sort();
        phenotypes.dedup();
        let unique_phenotypes = phenotypes.len();

        // Mean nuclear density (proxy: mean area of cells in module)
        let mean_density: f64 =
            module.iter().map(|&i| nodes[i].area.max(1e-12)).sum::<f64>() / module.len() as f64;

        let iq = if mean_density > 1e-18 {
            unique_phenotypes as f64 / mean_density
        } else {
            0.0
        };

        module_iqs.push(ModuleIQ {
            cell_indices: module.clone(),
            unique_phenotypes,
            mean_nuclear_density: mean_density,
            iq_score: iq,
        });
    }

    let n_modules = module_iqs.len();
    if n_modules == 0 {
        return TissueIQ {
            modules: vec![],
            mean_iq: 0.0,
            std_iq: 0.0,
            cv_iq: 0.0,
            heterogeneous_regions: vec![],
        };
    }

    let mean_iq: f64 = module_iqs.iter().map(|m| m.iq_score).sum::<f64>() / n_modules as f64;
    let var_iq: f64 = module_iqs
        .iter()
        .map(|m| {
            let d = m.iq_score - mean_iq;
            d * d
        })
        .sum::<f64>()
        / n_modules as f64;
    let std_iq = var_iq.sqrt();
    let cv_iq = if mean_iq > 1e-18 {
        std_iq / mean_iq
    } else {
        0.0
    };

    // Heterogeneous regions: modules with IQ > mean + 1 std
    let threshold = mean_iq + std_iq;
    let heterogeneous_regions: Vec<usize> = module_iqs
        .iter()
        .enumerate()
        .filter(|(_, m)| m.iq_score > threshold)
        .map(|(idx, _)| idx)
        .collect();

    TissueIQ {
        modules: module_iqs,
        mean_iq,
        std_iq,
        cv_iq,
        heterogeneous_regions,
    }
}

/// Convenience: run the full tissue IQ pipeline from raw nodes.
///
/// 1. Build kNN graph.
/// 2. Detect communities (target: √n modules or 8, whichever is larger).
/// 3. Compute tissue IQ.
pub fn tissue_iq_pipeline(
    nodes: &[SpatialNode],
    phenotype_fn: fn(&SpatialNode) -> u8,
) -> TissueIQ {
    let n = nodes.len();
    if n < 4 {
        return TissueIQ {
            modules: vec![],
            mean_iq: 0.0,
            std_iq: 0.0,
            cv_iq: 0.0,
            heterogeneous_regions: vec![],
        };
    }
    let target_modules = ((n as f64).sqrt() as usize).max(4).min(n / 2);
    let knn = build_knn_graph(nodes, DEFAULT_KNN_K);
    let communities = detect_communities(n, &knn, target_modules);
    compute_tissue_iq(nodes, &communities, phenotype_fn)
}

/// Default phenotype function: binarised by area (large → 1, small → 0).
pub fn default_area_phenotype(node: &SpatialNode) -> u8 {
    if node.area > 1.0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::SpatialNode;

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

    #[test]
    fn community_detection_merges_connected_cells() {
        // Two well-separated clusters
        let nodes: Vec<SpatialNode> = (0..6)
            .map(|i| {
                if i < 3 {
                    make_node(i, i as f64 * 2.0, 0.0, 1.0)
                } else {
                    make_node(i, 100.0 + i as f64 * 2.0, 0.0, 1.0)
                }
            })
            .collect();
        let knn = build_knn_graph(&nodes, 2);
        let communities = detect_communities(6, &knn, 2);

        assert_eq!(communities.len(), 2, "should find exactly 2 communities");
        // First 3 cells should be in one community, last 3 in the other
        let c0_has_first: bool = communities[0].contains(&0);
        let n_in_c0 = communities[0].len();
        let n_in_c1 = communities[1].len();
        assert!(n_in_c0 >= 2 && n_in_c1 >= 2, "each community should have ≥2 cells");
        // Cells within a cluster should stay together
        if c0_has_first {
            assert!(communities[0].contains(&1) || communities[0].contains(&2));
        }
    }

    #[test]
    fn tissue_iq_heterogeneous_modules_detected() {
        // Two modules: one uniform, one mixed.
        // Both have same average area so IQ differences reflect phenotype diversity.
        let mut nodes = Vec::new();
        // Module 0: uniform small cells (all phenotype 0)
        for i in 0..5 {
            nodes.push(make_node(i, i as f64, 0.0, 1.0));
        }
        // Module 1: mixed phenotypes with same average area
        for i in 5..10 {
            let area = if i % 2 == 0 { 0.3 } else { 1.7 };
            nodes.push(make_node(i, 100.0 + i as f64, 0.0, area));
        }

        let modules = vec![
            (0..5).collect::<Vec<usize>>(),
            (5..10).collect::<Vec<usize>>(),
        ];

        let iq = compute_tissue_iq(&nodes, &modules, default_area_phenotype);

        assert_eq!(iq.modules.len(), 2);
        // Module 1 should have more unique phenotypes than Module 0
        assert!(
            iq.modules[1].unique_phenotypes > iq.modules[0].unique_phenotypes,
            "mixed module should have more unique phenotypes"
        );
        // IQ should be valid for both
        assert!(iq.modules[0].iq_score > 0.0);
        assert!(iq.modules[1].iq_score > 0.0);
    }

    #[test]
    fn pipeline_returns_valid_structure() {
        let nodes: Vec<SpatialNode> = (0..20)
            .map(|i| {
                let area = if i < 10 { 0.5 } else { 2.0 };
                make_node(i, i as f64 * 2.0, (i % 5) as f64 * 2.0, area)
            })
            .collect();

        let iq = tissue_iq_pipeline(&nodes, default_area_phenotype);

        assert!(iq.modules.len() >= 2, "should detect multiple modules");
        assert!(iq.cv_iq >= 0.0, "CV should be non-negative");
        // Should have some heterogeneous regions
        assert!(!iq.heterogeneous_regions.is_empty() || iq.cv_iq < 0.01);
    }
}