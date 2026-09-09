//! Nexus-Forge Cyto **Core** — thin wrapper around the shared geometry engine.
//!
//! All geometry logic lives in `nexus_forge_cyto_geometry`. This crate adds
//! the CLI binary (`nexus-core-cli`) and a simplified orchestrator for pure-geometry
//! enrichment (no Mojo, no FFI, no mmap).

pub mod geometric_orchestrator;

// Re-export the shared geometry types so external callers see a flat API.
pub use nexus_forge_cyto_geometry::classifier::{evaluate_malignancy, malignancy_score, median_f64, CellFeatures};
pub use nexus_forge_cyto_geometry::spatial::{
    calculate_convex_hull, calculate_knn_density, calculate_rbf_field, default_rbf_sigma,
    distance_to_polygon, polygon_area_shoelace_abs, polygon_centroid_xy, polygon_circularity,
    polygon_eccentricity, polygon_perimeter, spatial_nodes_from_rings, ClinicalStatus, SpatialNode,
    DEFAULT_KNN_K,
};
pub use nexus_forge_cyto_geometry::TARGET_RING;

pub use crate::geometric_orchestrator::{
    format_geometric_cell_diagnosis, geometric_results_from_json_bytes,
    write_enriched_geometric_output, CytoCellResult, GeometricOrchestratorError,
};