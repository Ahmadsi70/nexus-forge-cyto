//! Geometric JSON coordinator: `(32,2)` contour envelopes -> kappa + spatial/tabular enrichment.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::classifier::{self, CellFeatures};
use crate::core::{
    expected_matrix_byte_len, CytoNeighborMatrixView, CytoStatus, CYTO_NEIGHBOR_ROW_COUNT,
};
use crate::mojo_bind::{invoke_mojo_curvature, mojo_work_f32_elems};
use crate::pipeline_parallel::CytoCellResult;
use crate::spatial::{self};
use nexus_forge_cyto_geometry::{
    config_space, constrained_opt, cyclic_fusion, multiscale, path_entropy, spectral, tissue_iq,
};

const DIMS_D: u32 = 2;
const TARGET_RING: usize = 128;   // Geometry: full-resolution ring
const KAPPA_RING: usize = 64;     // FFT κ engine: downsampled ring

// Compile-time guard: ring-based geometry uses exactly 2 spatial dimensions.
const _: () = assert!(DIMS_D == 2, "DIMS_D must be 2 for ring-based geometry");

/// Arc-length downsample 128 → 64 by taking every other point (Nyquist in spatial domain).
/// Preserves the closed-ring topology while halving the FFT workload.
fn downsample_128_to_64(ring: &[[f32; 2]; TARGET_RING]) -> [[f32; 2]; KAPPA_RING] {
    let mut out = [[0.0_f32; 2]; KAPPA_RING];
    for i in 0..KAPPA_RING {
        out[i] = ring[i * 2];
    }
    out
}

#[derive(Debug, Deserialize)]
pub(crate) struct GeometricResponse {
    pub cell_count: usize,
    pub cells: Vec<Vec<[f32; 2]>>,
}

#[inline]
fn ring_area_shoelace_px_sq(ring: &[[f32; 2]]) -> f32 {
    let n = ring.len();
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
        // NaN-safe total ordering: NaN pushed to the end so it never
        // silently compares as Equal to a finite area value.
        if a.is_nan() && b.is_nan() {
            std::cmp::Ordering::Equal
        } else if a.is_nan() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }));
    // Exclude trailing NaNs from the median computation.
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
    Mojo(CytoStatus),
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
            GeometricOrchestratorError::Mojo(s) => write!(
                f,
                "Mojo kappa engine reported {:?} - {}",
                s,
                s.operator_hint()
            ),
            GeometricOrchestratorError::Shape {
                cell_index,
                got_points,
            } => write!(
                f,
                "cell {cell_index}: expected {TARGET_RING} boundary samples per ring, got {got_points}"
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

fn mojo_curvature_for_ring(
    cell_id: u32,
    ring: &[[f32; 2]; TARGET_RING],
) -> Result<CytoCellResult, GeometricOrchestratorError> {
    // Downsample 128 → 64 for the κ FFT engine
    let ring64 = downsample_128_to_64(ring);

    let scratch_n = mojo_work_f32_elems(DIMS_D)
        .ok_or(GeometricOrchestratorError::Mojo(CytoStatus::CytoErrInvalidLen))?;
    let byte_len = expected_matrix_byte_len(DIMS_D)
        .ok_or(GeometricOrchestratorError::Mojo(CytoStatus::CytoErrInvalidLen))?;

    let mut matrix: Vec<f32> = Vec::with_capacity(KAPPA_RING * 2);
    for p in &ring64 {
        matrix.push(p[0]);
        matrix.push(p[1]);
    }

    let mut scratch = vec![0.0_f32; scratch_n];
    let mut kappa: f32 = 0.0;
    let mut clinical_raw: u32 = 0;

    let view = CytoNeighborMatrixView {
        data: matrix.as_ptr(),
        dims_d: DIMS_D,
        row_count: CYTO_NEIGHBOR_ROW_COUNT,
        stride_elems: DIMS_D,
        _implicit_pad_before_byte_length: 0,
        byte_length: byte_len,
        _align_reserved_tail: [0_u8; 32],
    };

    let mcode = unsafe {
        invoke_mojo_curvature(
            std::ptr::addr_of!(view),
            1.0_f32,
            std::ptr::addr_of_mut!(kappa),
            std::ptr::addr_of_mut!(clinical_raw),
            scratch.as_mut_ptr(),
            scratch.len(),
        )
    };
    let mstat = CytoStatus::from_u32_transport(mcode);
    if mstat != CytoStatus::CytoSuccess {
        return Err(GeometricOrchestratorError::Mojo(mstat));
    }

    let clinical = CytoStatus::from_u32_transport(clinical_raw);
    let mal = match clinical {
        CytoStatus::CytoClinicalMalignant => 1_u8,
        CytoStatus::CytoClinicalNormal => 0_u8,
        _ => {
            return Err(GeometricOrchestratorError::Mojo(
                CytoStatus::CytoErrMojoUnknownTransport,
            ));
        }
    };

    Ok(CytoCellResult {
        cell_id,
        kappa,
        is_malignant: mal,
        _pad: [0_u8; 7],
    })
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
        if ring_vec.len() != TARGET_RING {
            return Err(GeometricOrchestratorError::Shape {
                cell_index: idx,
                got_points: ring_vec.len(),
            });
        }
        let ring: Vec<[f32; 2]> = ring_vec.iter().copied().collect();
        areas.push(ring_area_shoelace_px_sq(&ring));
        rings.push(ring);
    }

    let median_area = median_f32(&areas);
    let malignancy_area_threshold = 1.5 * median_area;
    let mut out: Vec<CytoCellResult> = Vec::with_capacity(rings.len());

    for (idx, ring) in rings.iter().enumerate() {
        // Convert Vec<[f32;2]> to [[f32;2]; 128] for the kappa engine
        let mut ring128 = [[0.0_f32; 2]; TARGET_RING];
        for (j, p) in ring.iter().enumerate().take(TARGET_RING) {
            ring128[j] = *p;
        }
        let mut row = mojo_curvature_for_ring(idx as u32, &ring128)?;
        let a = areas[idx];
        if median_area > 1e-6 && a > malignancy_area_threshold {
            row.is_malignant = 1;
        }
        out.push(row);
    }
    Ok((out, rings))
}

/// Compute all 14 mathematical innovation mappings and return as a JSON value
/// to be embedded in the enriched output under the `innovations` key.
fn compute_innovation_layer(
    nodes: &[spatial::SpatialNode],
    knn_density: &[f64],
    rbf_field: &[f64],
    rows: &[CytoCellResult],
) -> serde_json::Value {
    use nexus_forge_cyto_geometry::spatial::SpatialNode as GeomNode;
    use nexus_forge_cyto_geometry::classifier::CellFeatures as GeomFeatures;
    use nexus_forge_cyto_geometry::spatial::ClinicalStatus as GeomClinical;

    let n = nodes.len();
    if n == 0 {
        return serde_json::json!({});
    }

    // Convert local types to geometry-crate types
    let geom_nodes: Vec<GeomNode> = nodes.iter().map(|nd| GeomNode {
        cell_id: nd.cell_id,
        centroid_x: nd.centroid_x,
        centroid_y: nd.centroid_y,
        area: nd.area,
        perimeter: nd.perimeter,
        circularity: nd.circularity,
        eccentricity: nd.eccentricity,
        solidity: 1.0,
        convexity: 1.0,
        aspect_ratio: 1.0,
        hu_moments: [0.0; 7],
        distance_to_tumor_edge: nd.distance_to_tumor_edge,
        clinical_status: nd.clinical_status.map(|cs| match cs {
            spatial::ClinicalStatus::Normal => GeomClinical::Normal,
            spatial::ClinicalStatus::Malignant => GeomClinical::Malignant,
        }),
    }).collect();

    // Build CellFeatures for spectral embedding (from converted nodes)
    let areas: Vec<f64> = geom_nodes.iter().map(|nd| nd.area).collect();
    let med_area = classifier::median_f64(&areas);
    let med_knn = classifier::median_f64(knn_density);

    let cell_features: Vec<GeomFeatures> = (0..n)
        .map(|i| {
            let nd = &geom_nodes[i];
            GeomFeatures {
                area: if med_area > 1e-18 { nd.area / med_area } else { 1.0 },
                circularity: nd.circularity,
                rbf_risk_score: rbf_field[i],
                knn_density: if med_knn > 1e-18 { knn_density[i] / med_knn } else { 1.0 },
                eccentricity: nd.eccentricity,
            }
        })
        .collect();

    // 1. Spectral embedding (Mapping #1)
    let embedding = spectral::spectral_decompose(&cell_features);
    let ortho_pairs = spectral::orthogonal_cell_pairs(&embedding, 0.05);
    let clusters = spectral::spectral_clusters(&embedding);

    // 2. Density ratio graph (Mapping #2)
    use nexus_forge_cyto_geometry::spatial as geom_spatial;
    let knn_edges = geom_spatial::build_knn_graph(&geom_nodes, geom_spatial::DEFAULT_KNN_K);
    let density_edges =
        geom_spatial::compute_density_ratios(&geom_nodes, &knn_edges, geom_spatial::default_cell_density);
    let density_stats = geom_spatial::density_ratio_summary(&density_edges);

    // 3. Tissue IQ (Mapping #3)
    let iq = tissue_iq::tissue_iq_pipeline(&geom_nodes, tissue_iq::default_area_phenotype);

    // 5. RBF tissue surface (Mapping #5) — sampled at coarse grid
    let xs: Vec<f64> = geom_nodes.iter().map(|nd| nd.centroid_x).collect();
    let ys: Vec<f64> = geom_nodes.iter().map(|nd| nd.centroid_y).collect();
    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let rbf_config = geom_spatial::RBFConfig::default();
    let surface = geom_spatial::rbf_tissue_surface(
        &geom_nodes, (x_min, x_max, y_min, y_max), 16, &rbf_config,
    );

    // 8. Harmonic gradient (Mapping #8) along x-axis
    let positions: Vec<[f64; 2]> = geom_nodes.iter().map(|nd| [nd.centroid_x, nd.centroid_y]).collect();
    let gradient = geom_spatial::fit_harmonic_gradient(&positions, &areas, &[1.0, 0.0]);

    // 7. Configuration space (Mapping #7)
    let kappas: Vec<f32> = rows.iter().map(|r| r.kappa).collect();
    let cfg_space = config_space::ConfigurationSpace::from_nodes(&geom_nodes, &kappas);
    let cfg_entropy = config_space::configuration_entropy(&cfg_space);
    let orient_coherence = config_space::local_orientation_coherence(&cfg_space, 8);

    // 10. Multi-scale bridge (Mapping #10, #14)
    let bridge = multiscale::compute_two_scale_bridge(&geom_nodes);
    let tiers = multiscale::ScaleTiers::default();

    // 10b. Bridge-normalised features (Mapping #10 extension — unused helpers)
    let area_f64: Vec<f64> = geom_nodes.iter().map(|nd| nd.area).collect();
    let norm_areas = multiscale::normalise_by_bridge(&area_f64, &bridge);
    let norm_area_mean = norm_areas.iter().sum::<f64>() / norm_areas.len().max(1) as f64;
    let norm_area_std = {
        let mu = norm_area_mean;
        (norm_areas.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / norm_areas.len().max(1) as f64).sqrt()
    };

    // 14b. Multi-scale features at all four tiers (Mapping #14 extension)
    let median_area = if !areas.is_empty() {
        let mut sorted = areas.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[areas.len() / 2]
    } else { 0.0 };
    let ms_features = multiscale::multiscale_features(median_area, tiers.nuclear, &tiers);

    // 11. Optimal embedding axis (Mapping #11)
    let dist_deg: Vec<[f64; 2]> = knn_density
        .iter()
        .zip(&areas)
        .map(|(&d, &a)| [d, a])
        .collect();
    let clin_labels: Vec<GeomClinical> = rows
        .iter()
        .map(|r| {
            if r.is_malignant != 0 {
                GeomClinical::Malignant
            } else {
                GeomClinical::Normal
            }
        })
        .collect();
    let opt_proj = multiscale::grid_search_embedding_axis(&dist_deg, &clin_labels, 7);

    // 12. Path entropy (Mapping #12) — radial from first cell
    let radial_path = path_entropy::radial_geodesic_path(&geom_nodes, 0, n.min(50));
    let path_ent = path_entropy::compute_path_entropy(
        &geom_nodes, &radial_path, |nd| nd.area.max(1e-12),
    );

    // 13. Equilibrium patches (Mapping #13)
    let eq_patches = path_entropy::detect_equilibrium_patches(
        &geom_nodes, &knn_edges, |nd| nd.area.max(1e-12), 0.5,
    );

    // 4. Constrained optimization (Mapping #4) — lightweight, on a small subgraph
    let sub_n = n.min(30);
    let sub_features: Vec<Vec<f64>> = (0..sub_n).map(|i| vec![areas[i], knn_density[i]]).collect();
    let sub_edges: Vec<(usize, usize)> = knn_edges
        .iter()
        .filter(|&&(a, b)| a < sub_n && b < sub_n)
        .copied()
        .collect();
    let constraints = constrained_opt::ConstrainedManifold::default();
    let init_w = vec![1.0_f64; sub_edges.len()];
    let (_opt_w, opt_loss, opt_iters) = constrained_opt::projected_gradient_descent(
        &init_w, &sub_edges, sub_n, &constraints, 0.001, 200, 1e-6,
        |w| constrained_opt::laplacian_smoothness_loss(w, &sub_edges, &sub_features),
    );

    serde_json::json!({
        // Mapping #1
        "spectral_embedding": {
            "singular_values": embedding.singular_values,
            "effective_rank_90": embedding.effective_rank_90,
            "frobenius_energy_ratio": embedding.frobenius_energy_ratio,
            "orthogonal_pair_count": ortho_pairs.len(),
            "spectral_clusters": clusters,
        },
        // Mapping #2
        "density_ratios": {
            "n_edges": density_stats.n_edges,
            "mean_ratio": density_stats.mean_ratio,
            "std_ratio": density_stats.std_ratio,
            "phi_like_count": density_stats.phi_like_count,
            "pi_like_count": density_stats.pi_like_count,
            "balanced_count": density_stats.balanced_count,
            "alignment_fraction": density_stats.alignment_fraction,
        },
        // Mapping #3
        "tissue_iq": {
            "n_modules": iq.modules.len(),
            "mean_iq": iq.mean_iq,
            "std_iq": iq.std_iq,
            "cv_iq": iq.cv_iq,
            "heterogeneous_region_count": iq.heterogeneous_regions.len(),
            "max_module_iq": iq.modules.iter().map(|m| m.iq_score).fold(0.0_f64, f64::max),
        },
        // Mapping #4
        "constrained_optimization": {
            "n_optimized_edges": sub_edges.len(),
            "final_loss": opt_loss,
            "iterations": opt_iters,
        },
        // Mapping #5
        "rbf_surface": {
            "resolution": 16,
            "min_value": surface.iter().flatten().cloned().fold(f64::INFINITY, f64::min),
            "max_value": surface.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max),
        },
        // Mapping #7
        "configuration_space": {
            "total_dim": cfg_space.total_dim,
            "entropy": cfg_entropy,
            "local_orientation_coherence_k8": orient_coherence,
        },
        // Mapping #8
        "harmonic_gradient": match &gradient {
            Some(g) => serde_json::json!({
                "amplitude": g.amplitude,
                "frequency": g.frequency,
                "exponential_decay": g.exponential_decay,
            }),
            None => serde_json::json!(null),
        },
        // Mapping #10
        "two_scale_bridge": {
            "rho_macro": bridge.rho_macro,
            "rho_micro": bridge.rho_micro,
            "phi_Qg": bridge.phi_qg,
            "lambda": bridge.lambda,
            "geometric_mean": bridge.geometric_mean,
        },
        // Mapping #11
        "optimal_projection": {
            "best_theta_rad": opt_proj.best_theta,
            "best_alpha": opt_proj.best_alpha,
            "best_beta": opt_proj.best_beta,
            "separation_score": opt_proj.separation_score,
        },
        // Mapping #12
        "path_entropy": {
            "path_length": radial_path.len(),
            "total_entropy": path_ent.total_entropy,
            "max_entropy_delta": path_ent.entropy_delta.iter().cloned().fold(0.0_f64, f64::max),
            "final_cumulative_risk": path_ent.cumulative_risk.last().copied().unwrap_or(0.0),
        },
        // Mapping #13
        "equilibrium_patches": {
            "n_patches": eq_patches.len(),
            "largest_patch_size": eq_patches.iter().map(|p| p.len()).max().unwrap_or(0),
        },
        // Mapping #14
        "tiered_scales": {
            "organelle_to_nuclear": tiers.scale_ratio(tiers.organelle, tiers.nuclear),
            "nuclear_to_cellular": tiers.scale_ratio(tiers.nuclear, tiers.cellular),
            "cellular_to_neighborhood": tiers.scale_ratio(tiers.cellular, tiers.neighborhood),
            // Mapping #14b: same feature (median area) re-expressed at each tier
            "median_area_at_organelle_tier": ms_features[0],
            "median_area_at_nuclear_tier": ms_features[1],
            "median_area_at_cellular_tier": ms_features[2],
            "median_area_at_neighborhood_tier": ms_features[3],
        },
        // Mapping #10b: bridge-normalised area (cross-slide comparable)
        "bridge_normalised_area": {
            "mean": norm_area_mean,
            "std": norm_area_std,
            "first_cell": norm_areas.get(0).copied().unwrap_or(0.0),
        },
    })
}

/// Compute cyclic fusion (FFT ring convolution) layer — Mapping #9.
///
/// Uses the 64-point downsampled rings to compute:
/// - Angular power spectra for each cell
/// - Spectral ring distances between cell pairs
/// - Low-pass filtered (smoothed) rings
/// - Cyclic convolution of representative cell pairs
fn compute_cyclic_fusion_layer(rings: &[Vec<[f32; 2]>]) -> serde_json::Value {
    const N64: usize = 64;

    if rings.len() < 2 {
        return serde_json::json!({});
    }

    // Downsample each 128-point ring to 64 points for the FFT engine
    let mut rings64: Vec<[[f32; 2]; N64]> = Vec::with_capacity(rings.len());
    for ring in rings {
        let mut r64 = [[0.0_f32; 2]; N64];
        for (j, p) in ring.iter().enumerate().take(N64) {
            r64[j] = *p;
        }
        rings64.push(r64);
    }

    // Angular power spectra for all cells
    let spectra: Vec<[f32; N64]> = rings64
        .iter()
        .map(|r| cyclic_fusion::angular_spectrum(r))
        .collect();

    // Frequency-domain distance: first few cells vs all others
    let n_pairs = rings.len().min(10);
    let mut pair_distances: Vec<serde_json::Value> = Vec::with_capacity(n_pairs * n_pairs);
    for i in 0..n_pairs {
        for j in (i + 1)..n_pairs {
            let d = cyclic_fusion::spectral_ring_distance(&rings64[i], &rings64[j], 8);
            pair_distances.push(serde_json::json!({
                "cell_a": i,
                "cell_b": j,
                "spectral_distance_8freq": d,
            }));
        }
    }

    // Low-pass filter (smooth) the first ring
    let smoothed = cyclic_fusion::low_pass_ring(&rings64[0], 4);
    let smoothed_flat: Vec<f64> = smoothed.iter().map(|p| p[0] as f64).collect();

    // Cyclic convolution of first two cells
    let conv_ring = cyclic_fusion::cyclic_convolution_rings(&rings64[0], &rings64[1 % rings.len()]);
    let conv_flat: Vec<f64> = conv_ring.iter().map(|p| p[0] as f64).collect();

    // Mean spectral energy per cell (DC component excluded)
    let mean_spectral_energy: Vec<f64> = spectra
        .iter()
        .map(|spec| spec[1..].iter().map(|&v| v as f64).sum::<f64>() / (N64 - 1) as f64)
        .collect();

    serde_json::json!({
        "n_cells_analyzed": rings.len(),
        "n_spectra_computed": spectra.len(),
        "pairwise_spectral_distances": pair_distances,
        "mean_spectral_energy_per_cell": mean_spectral_energy,
        "smoothed_ring_cell_0_x_coords": smoothed_flat,
        "convolution_ring_c0_c1_x_coords": conv_flat,
    })
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
    image_path: Option<&Path>,
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
    let mut all_features = Vec::with_capacity(n_cells);

    for i in 0..n_cells {
        let node = &spatial_nodes[i];
        let rbf = spatial::calculate_rbf_field(node, &spatial_nodes, rbf_sigma);
        // Store the *normalized* RBF risk score so the JSON output (spatial_features)
        // matches the value fed into the classifier — not the raw sum.
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
        // Merge: keep Mojo/area label if already Malignant; otherwise apply
        // heuristic classifier so that spatial features can upgrade Normal → Malignant.
        let heuristic_mal = matches!(
            classifier::evaluate_malignancy(&features),
            spatial::ClinicalStatus::Malignant
        );
        if heuristic_mal {
            rows[i].is_malignant = 1;
        }
        
        all_features.push(features);
    }
    
    // --- GNN Inference (Phase 1 AI Integration) ---
    let visual_features = if let Some(img_path) = image_path {
        match crate::vision::extract_visual_features(img_path, &spatial_nodes) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Vision extraction failed: {:?}", e);
                None
            }
        }
    } else {
        None
    };

    // ----------------------------------------------

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
        // JSON/schema require finite non-negative distances; empty hull ⇒ no tumor edge.
        node.distance_to_tumor_edge = if raw_dist.is_finite() {
            raw_dist
        } else {
            0.0
        };
    }

    let mut spatial_features: Vec<serde_json::Value> = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let node = &spatial_nodes[i];
        let mut obj = serde_json::json!({
            "cell_id": node.cell_id,
            "centroid": [node.centroid_x, node.centroid_y],
            "knn_density": knn_density[i],
            "rbf_risk_score": rbf_field[i],
            "area": node.area,
            "perimeter": node.perimeter,
            "circularity": node.circularity,
            "eccentricity": node.eccentricity,
            "distance_to_tumor_edge": node.distance_to_tumor_edge,
        });
        
        if let Some(v_feats) = &visual_features {
            let feat_row: Vec<f64> = v_feats.row(i).to_vec();
            obj.as_object_mut().unwrap().insert("vision_features".to_string(), serde_json::json!(feat_row));
        }
        spatial_features.push(obj);
    }

    let mut enriched: serde_json::Value =
        serde_json::from_slice(raw).map_err(|e| GeometricOrchestratorError::Request(e.to_string()))?;

    // ── Innovation layer: 14 mathematical mappings ──
    let mut innovations = compute_innovation_layer(&spatial_nodes, &knn_density, &rbf_field, &rows);

    // ── Cyclic fusion (FFT ring convolution) — Mapping #9 ──
    // Uses the 64-point downsampled rings to compute spectral ring distances,
    // angular spectra, and cyclic fusion between cell pairs.
    if let Some(innov_obj) = innovations.as_object_mut() {
        innov_obj.insert(
            "cyclic_fusion_fft".to_string(),
            compute_cyclic_fusion_layer(&rings),
        );
    }

    let obj = enriched.as_object_mut().ok_or_else(|| {
        GeometricOrchestratorError::Bridge(
            "segment JSON root must be a JSON object to attach clinical labels".into(),
        )
    })?;
    obj.insert("innovations".to_string(), innovations);
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
