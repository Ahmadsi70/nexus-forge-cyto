//! Geometric JSON coordinator: `(32,2)` contour envelopes → spatial/tabular enrichment.
//!
//! This is the **pure Rust geometry core** — no Mojo, no FFI, no mmap.
//! Identical inputs ⇒ identical outputs, bit-for-bit.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use nexus_forge_cyto_geometry::classifier::{self, CellFeatures};
use nexus_forge_cyto_geometry::spatial::{self};

const DIMS_D: u32 = 2;

// Compile-time guard: ring-based geometry uses exactly 2 spatial dimensions.
const _: () = assert!(DIMS_D == 2, "DIMS_D must be 2 for ring-based geometry");

/// Per-cell output payload for downstream analytics (Phase-4 batch export).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct CytoCellResult {
    pub cell_id: u32,
    pub kappa: f32,
    pub is_malignant: u8,
    pub _pad: [u8; 7],
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeometricResponse {
    pub cell_count: usize,
    pub cells: Vec<Vec<[f32; 2]>>,
}

#[inline]
fn ring_area_shoelace_px_sq(ring: &[[f32; 2]]) -> f32 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        let x1 = ring[i][0] as f64;
        let y1 = ring[i][1] as f64;
        let x2 = ring[(i + 1) % n][0] as f64;
        let y2 = ring[(i + 1) % n][1] as f64;
        acc += x1 * y2 - x2 * y1;
    }
    (0.5 * acc.abs()) as f32
}

fn median_f32(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| {
        if a.is_nan() && b.is_nan() {
            std::cmp::Ordering::Equal
        } else if a.is_nan() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }));
    let finite_count = v.iter().position(|&x| x.is_nan()).unwrap_or(v.len());
    if finite_count == 0 {
        return 0.0;
    }
    let mid = finite_count / 2;
    if finite_count % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) * 0.5
    }
}

#[derive(Debug)]
pub enum GeometricOrchestratorError {
    WriteSegmentJson {
        path: PathBuf,
        source: io::Error,
    },
    Request(String),
    Shape {
        cell_index: usize,
        got_points: usize,
    },
    Bridge(String),
}

impl fmt::Display for GeometricOrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometricOrchestratorError::WriteSegmentJson { path, source } => write!(
                f,
                "failed to write enriched segment JSON to {}: {source}",
                path.display()
            ),
            GeometricOrchestratorError::Request(s) => {
                write!(f, "segment JSON deserialize failed: {s}")
            }
            GeometricOrchestratorError::Shape {
                cell_index,
                got_points,
            } => write!(
                f,
                "cell {cell_index}: expected 32 boundary samples per ring, got {got_points}"
            ),
            GeometricOrchestratorError::Bridge(s) => write!(
                f,
                "segment envelope / clinical attach rejected: {s}"
            ),
        }
    }
}

impl std::error::Error for GeometricOrchestratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GeometricOrchestratorError::WriteSegmentJson { source, .. } => Some(source),
            _ => None,
        }
    }
}

type GeometricRingBatch = (Vec<CytoCellResult>, Vec<Vec<[f32; 2]>>);

fn geometric_results_and_rings_from_response(
    vr: &GeometricResponse,
) -> Result<GeometricRingBatch, GeometricOrchestratorError> {
    if vr.cells.len() != vr.cell_count {
        return Err(GeometricOrchestratorError::Bridge(format!(
            "cell_count {} != cells.len() {}",
            vr.cell_count,
            vr.cells.len()
        )));
    }

    let mut rings: Vec<Vec<[f32; 2]>> = Vec::with_capacity(vr.cells.len());
    let mut areas: Vec<f32> = Vec::with_capacity(vr.cells.len());

    for (idx, ring_vec) in vr.cells.iter().enumerate() {
        areas.push(ring_area_shoelace_px_sq(ring_vec));
        rings.push(ring_vec.clone());
    }

    let median_area = median_f32(&areas);
    let malignancy_area_threshold = 1.5 * median_area;
    let mut out: Vec<CytoCellResult> = Vec::with_capacity(rings.len());

    for (idx, _ring) in rings.iter().enumerate() {
        let a = areas[idx];
        // Pure geometry core: malignancy is determined solely by area heuristic
        // and downstream spatial classifier — no Mojo κ engine.
        let is_mal = if median_area > 1e-6 && a > malignancy_area_threshold {
            1_u8
        } else {
            0_u8
        };
        out.push(CytoCellResult {
            cell_id: idx as u32,
            kappa: 0.0, // not computed in pure geometry core
            is_malignant: is_mal,
            _pad: [0_u8; 7],
        });
    }
    Ok((out, rings))
}

pub fn geometric_results_from_json_bytes(
    raw: &[u8],
) -> Result<Vec<CytoCellResult>, GeometricOrchestratorError> {
    let vr: GeometricResponse =
        serde_json::from_slice(raw).map_err(|e| GeometricOrchestratorError::Request(e.to_string()))?;
    geometric_results_and_rings_from_response(&vr).map(|(rows, _)| rows)
}

pub fn write_enriched_geometric_output(
    raw: &[u8],
    output_json: &Path,
) -> Result<Vec<CytoCellResult>, GeometricOrchestratorError> {
    let vr: GeometricResponse =
        serde_json::from_slice(raw).map_err(|e| GeometricOrchestratorError::Request(e.to_string()))?;
    let (mut rows, rings) = geometric_results_and_rings_from_response(&vr)?;

    let mut spatial_nodes = spatial::spatial_nodes_from_rings(rings.as_slice(), None);
    let knn_density = spatial::calculate_knn_density(&spatial_nodes, spatial::DEFAULT_KNN_K);
    let rbf_sigma = spatial::default_rbf_sigma(&knn_density);
    let n_cells = spatial_nodes.len();
    let denom_neighbors = n_cells.saturating_sub(1).max(1) as f64;
    let areas: Vec<f64> = spatial_nodes.iter().map(|n| n.area).collect();
    let med_area = classifier::median_f64(&areas);
    let med_knn = classifier::median_f64(&knn_density);
    let mut rbf_field = vec![0.0_f64; n_cells];

    for i in 0..n_cells {
        let node = &spatial_nodes[i];
        let rbf = spatial::calculate_rbf_field(node, &spatial_nodes, rbf_sigma);
        // Store the *normalized* RBF risk score so the JSON output matches
        // the value fed into the classifier.
        let rbf_norm = rbf / denom_neighbors;
        rbf_field[i] = rbf_norm;
        let area_ratio = if med_area > 1e-18 { node.area / med_area } else { 1.0 };
        let knn_ratio = if med_knn > 1e-18 { knn_density[i] / med_knn } else { 1.0 };
        let features = CellFeatures {
            area: area_ratio,
            circularity: node.circularity,
            rbf_risk_score: rbf_norm,
            knn_density: knn_ratio,
            eccentricity: node.eccentricity,
        };
        // Merge: keep area label if already Malignant; otherwise apply
        // heuristic classifier so that spatial features can upgrade Normal → Malignant.
        let heuristic_mal = matches!(
            classifier::evaluate_malignancy(&features),
            spatial::ClinicalStatus::Malignant
        );
        if heuristic_mal {
            rows[i].is_malignant = 1;
        }
    }

    let malignant_centroids: Vec<[f64; 2]> = (0..n_cells)
        .filter(|&i| rows[i].is_malignant != 0)
        .map(|i| [spatial_nodes[i].centroid_x, spatial_nodes[i].centroid_y])
        .collect();
    let tumor_hull = spatial::calculate_convex_hull(&malignant_centroids);
    for node in spatial_nodes.iter_mut().take(n_cells) {
        let raw_dist = spatial::distance_to_polygon(
            [node.centroid_x, node.centroid_y],
            &tumor_hull,
        );
        node.distance_to_tumor_edge = if raw_dist.is_finite() {
            raw_dist
        } else {
            0.0
        };
    }

    let mut spatial_features: Vec<serde_json::Value> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let node = &spatial_nodes[i];
        spatial_features.push(serde_json::json!({
            "cell_id": node.cell_id,
            "centroid": [node.centroid_x, node.centroid_y],
            "knn_density": knn_density[i],
            "rbf_risk_score": rbf_field[i],
            "area": node.area,
            "perimeter": node.perimeter,
            "circularity": node.circularity,
            "eccentricity": node.eccentricity,
            "solidity": node.solidity,
            "convexity": node.convexity,
            "aspect_ratio": node.aspect_ratio,
            "hu_moments": node.hu_moments,
            "distance_to_tumor_edge": node.distance_to_tumor_edge,
        }));
    }

    let mut enriched: serde_json::Value =
        serde_json::from_slice(raw).map_err(|e| GeometricOrchestratorError::Request(e.to_string()))?;
    let clinical: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::Value::String(if r.is_malignant != 0 {
                "Malignant".into()
            } else {
                "Normal".into()
            })
        })
        .collect();
    let obj = enriched.as_object_mut().ok_or_else(|| {
        GeometricOrchestratorError::Bridge(
            "segment JSON root must be a JSON object to attach clinical labels".into(),
        )
    })?;
    obj.insert("clinical".to_string(), serde_json::Value::Array(clinical));
    obj.insert(
        "spatial_analysis".to_string(),
        serde_json::json!({
            "knn_k": spatial::DEFAULT_KNN_K,
            "rbf_sigma": rbf_sigma,
            "knn_metric": "mean_euclidean_distance_to_k_neighbors",
        }),
    );
    obj.insert(
        "spatial_features".to_string(),
        serde_json::Value::Array(spatial_features),
    );

    let rendered = serde_json::to_vec_pretty(&enriched)
        .map_err(|e| GeometricOrchestratorError::Request(e.to_string()))?;
    std::fs::write(output_json, &rendered).map_err(|source| GeometricOrchestratorError::WriteSegmentJson {
        path: output_json.to_path_buf(),
        source,
    })?;
    Ok(rows)
}

pub fn format_geometric_cell_diagnosis(r: &CytoCellResult) -> String {
    let label = if r.is_malignant != 0 {
        "Malignant"
    } else {
        "Normal"
    };
    format!("cell_id={} kappa={:.6} clinical={}", r.cell_id, r.kappa, label)
}