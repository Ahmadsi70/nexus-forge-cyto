//! Nexus-Forge Cyto **Geometry** — deterministic spatial morphometrics & explainable scoring.
//!
//! This is the shared geometry engine used by all Nexus-Forge crates.
//! **No AI, no Mojo, no FFI, no mmap.** Just polygons → Rust math → enriched JSON.
//!
//! Identical inputs ⇒ identical outputs, bit-for-bit.

pub mod classifier;
pub mod config_space;
pub mod constrained_opt;
pub mod cyclic_fusion;
pub mod logging;
pub mod multiscale;
pub mod path_entropy;
pub mod spatial;
pub mod spectral;
pub mod tissue_iq;

// --- Re-export the public surface ---

pub use crate::classifier::{evaluate_malignancy, malignancy_score, median_f64, CellFeatures};
pub use crate::spatial::{
    calculate_convex_hull, calculate_knn_density, calculate_rbf_field, default_rbf_sigma,
    distance_to_polygon, polygon_area_shoelace_abs, polygon_centroid_xy, polygon_circularity,
    polygon_eccentricity, polygon_perimeter, spatial_nodes_from_rings, ClinicalStatus, SpatialNode,
    DEFAULT_KNN_K,
};
pub use crate::spectral::{spectral_decompose, orthogonal_cell_pairs, spectral_clusters, SpectralEmbedding, OrthogonalCellPair};
pub use crate::tissue_iq::{tissue_iq_pipeline, compute_tissue_iq, detect_communities, default_area_phenotype, TissueIQ, ModuleIQ};
pub use crate::multiscale::{compute_two_scale_bridge, normalise_by_bridge, tiered_normalize, multiscale_features, ScaleTiers, TwoScaleModel, OptimalProjection};
pub use crate::path_entropy::{compute_path_entropy, radial_geodesic_path, detect_equilibrium_patches, PathEntropy};
pub use crate::config_space::{ConfigurationSpace, configuration_entropy, local_orientation_coherence, quaternion_geodesic_distance};
pub use crate::constrained_opt::{ConstrainedManifold, projected_gradient_descent, laplacian_smoothness_loss, project_onto_feasible_set, projected_gradient_step};
pub use crate::cyclic_fusion::{cyclic_convolution_rings, angular_spectrum, spectral_ring_distance, low_pass_ring};

/// Fixed ring length (vertices per cell) — the universal contract across the pipeline.
pub const TARGET_RING: usize = 32;