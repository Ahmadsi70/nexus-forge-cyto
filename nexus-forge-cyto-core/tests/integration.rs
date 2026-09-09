//! Integration test: deterministic malignancy classification with realistic shapes.
//!
//! Verifies that the pure-geometry engine correctly distinguishes a benign-like
//! (small, round, well-separated) cell cohort from a malignant-like (large,
//! irregular, densely packed) cohort — using only Rust math, no AI.

use nexus_forge_cyto_core::{
    malignancy_score, polygon_circularity, polygon_perimeter, write_enriched_geometric_output,
    CellFeatures, ClinicalStatus,
};
use std::path::Path;

/// Build a JSON envelope from a list of 32-vertex rings.
fn envelope(rings: &[Vec<[f32; 2]>]) -> Vec<u8> {
    let payload = serde_json::json!({
        "cell_count": rings.len(),
        "cells": rings,
    });
    serde_json::to_vec(&payload).unwrap()
}

/// 32-vertex ring approximating a circle of radius `r` at `(cx, cy)`.
fn circle_ring(cx: f32, cy: f32, r: f32) -> Vec<[f32; 2]> {
    (0..32)
        .map(|i| {
            let t = (i as f32) / 32.0 * std::f32::consts::TAU;
            [cx + r * t.cos(), cy + r * t.sin()]
        })
        .collect()
}

/// 32-vertex ring approximating an elongated, irregular star (high eccentricity,
/// low circularity) — a malignant-like shape proxy.
fn irregular_ring(cx: f32, cy: f32, scale: f32) -> Vec<[f32; 2]> {
    (0..32)
        .map(|i| {
            let t = (i as f32) / 32.0 * std::f32::consts::TAU;
            // Modulate radius with a 3-lobe pattern → star-like irregularity.
            let rr = scale * (1.0 + 0.6 * (3.0 * t).cos());
            [cx + rr * t.cos() * 1.8, cy + rr * t.sin() * 0.6]
        })
        .collect()
}

#[test]
fn benign_round_small_cells_score_low() {
    // Five well-separated small round cells.
    let rings = vec![
        circle_ring(0.0, 0.0, 3.0),
        circle_ring(50.0, 0.0, 3.0),
        circle_ring(0.0, 50.0, 3.0),
        circle_ring(50.0, 50.0, 3.0),
        circle_ring(25.0, 25.0, 3.2),
    ];
    let raw = envelope(&rings);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("benign.json");
    let rows = write_enriched_geometric_output(&raw, &out).unwrap();

    let n_mal = rows.iter().filter(|r| r.is_malignant != 0).count();
    // With near-identical small round cells, the area gate (1.5× median) and the
    // spatial classifier should keep the overwhelming majority Normal.
    assert!(
        n_mal <= 1,
        "benign cohort flagged {n_mal} malignant — expected ≤ 1"
    );
}

#[test]
fn malignant_irregular_dense_cells_score_high() {
    // One clearly large, irregular, elongated cell among small round neighbors.
    let rings = vec![
        circle_ring(0.0, 0.0, 3.0),
        circle_ring(8.0, 0.0, 3.0),
        circle_ring(0.0, 8.0, 3.0),
        irregular_ring(4.0, 4.0, 5.0), // the malignant-like outlier
    ];
    let raw = envelope(&rings);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("mal.json");
    let rows = write_enriched_geometric_output(&raw, &out).unwrap();

    let n_mal = rows.iter().filter(|r| r.is_malignant != 0).count();
    assert!(n_mal >= 1, "expected at least one flagged malignant cell");
}

#[test]
fn malignancy_score_monotonic_with_eccentricity() {
    // Holding other features benign, increasing eccentricity should not decrease
    // the score (eccentricity is a positive-direction contributor).
    let base = CellFeatures {
        area: 1.0,
        circularity: 0.95,
        rbf_risk_score: 0.0,
        knn_density: 1.0,
        eccentricity: 0.0,
    };
    let low = malignancy_score(&CellFeatures { eccentricity: 0.1, ..base });
    let high = malignancy_score(&CellFeatures { eccentricity: 0.9, ..base });
    assert!(high >= low, "score must not drop as eccentricity rises");
}

#[test]
fn output_schema_has_all_expected_keys() {
    let rings = vec![circle_ring(0.0, 0.0, 4.0), circle_ring(30.0, 30.0, 4.0)];
    let raw = envelope(&rings);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("schema.json");
    write_enriched_geometric_output(&raw, &out).unwrap();

    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    for key in ["cell_count", "cells", "clinical", "spatial_analysis", "spatial_features"] {
        assert!(written.get(key).is_some(), "missing key: {key}");
    }
    // spatial_features entries must carry the full metric set.
    let sf = written["spatial_features"].as_array().unwrap();
    for cell in sf {
        for k in [
            "cell_id",
            "centroid",
            "area",
            "perimeter",
            "circularity",
            "eccentricity",
            "knn_density",
            "rbf_risk_score",
            "distance_to_tumor_edge",
        ] {
            assert!(cell.get(k).is_some(), "spatial_feature missing {k}");
        }
    }
}

#[test]
fn determinism_identical_inputs_identical_outputs() {
    let rings = vec![circle_ring(0.0, 0.0, 5.0), circle_ring(20.0, 0.0, 5.0)];
    let raw = envelope(&rings);

    let d1 = tempfile::tempdir().unwrap();
    let o1 = d1.path().join("a.json");
    write_enriched_geometric_output(&raw, &o1).unwrap();

    let d2 = tempfile::tempdir().unwrap();
    let o2 = d2.path().join("b.json");
    write_enriched_geometric_output(&raw, &o2).unwrap();

    assert_eq!(std::fs::read(&o1).unwrap(), std::fs::read(&o2).unwrap());
}

#[test]
fn polygon_perimeter_and_circularity_unit_square() {
    let sq = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let p = polygon_perimeter(&sq);
    let a = 1.0_f64;
    let c = polygon_circularity(a, p);
    assert!((p - 4.0).abs() < 1e-12);
    let expected = std::f64::consts::PI / 4.0; // unit disk vs unit square compactness
    assert!((c - expected).abs() < 1e-12);
}
