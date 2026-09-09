//! Mapping #1: SVD spectral embedding of the cell×feature incidence matrix.
//!
//! **Mathematical origin** (from `discovered_mathematical_mappings.md`):
//! > Sparse matrix **M** with shape 6236×1651 … dominant singular values …
//! > effective_rank_90_variance … orthogonal pairs.
//!
//! **Cell-network analogue:**
//! - Treat cells as rows and morphometric features as columns of M.
//! - Use SVD for cluster density, niche boundaries, and low-dimensional manifold
//!   coordinates for multiplexed membrane markers.
//! - Zero-dot-product pairs mirror decorrelated cell niches — useful for
//!   block-diagonal priors in graph neural nets.
//!
//! **Algorithm:**
//! - Build M of shape (n_cells × n_features) from [`CellFeatures`].
//! - Compute C = MᵀM (small d×d matrix, d ≤ 5).
//! - Eigendecompose C via power iteration + deflation.
//! - Recover singular values σ_i = √λ_i and left vectors u_i = M v_i / σ_i.
//! - Compute effective rank at 90% variance.
//! - Identify orthogonal cell pairs via u_i · u_j ≈ 0.

use crate::classifier::CellFeatures;

/// Result of spectral decomposition of the cell×feature matrix.
#[derive(Clone, Debug)]
pub struct SpectralEmbedding {
    /// Singular values in descending order (σ₁ ≥ σ₂ ≥ … ≥ σ_d).
    pub singular_values: Vec<f64>,
    /// Left singular vectors (cell embeddings). Each row corresponds to a cell,
    /// each column to a spectral component. Shape: (n_cells × rank).
    pub left_vectors: Vec<Vec<f64>>,
    /// Right singular vectors (feature loadings). Shape: (rank × n_features).
    pub right_vectors: Vec<Vec<f64>>,
    /// Effective rank: number of singular values needed to capture ≥90% of Frobenius energy.
    pub effective_rank_90: usize,
    /// Ratio of top-k Frobenius energy to total energy.
    pub frobenius_energy_ratio: f64,
}

/// Pairs of cells whose embedding vectors are approximately orthogonal.
#[derive(Clone, Debug)]
pub struct OrthogonalCellPair {
    pub cell_a: usize,
    pub cell_b: usize,
    pub dot_product: f64,
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Build the cell×feature incidence matrix.
///
/// Each cell contributes one row with the following features:
/// `[area, circularity, rbf_risk_score, knn_density, eccentricity]`.
pub fn build_incidence_matrix(features: &[CellFeatures]) -> Vec<Vec<f64>> {
    features
        .iter()
        .map(|f| vec![f.area, f.circularity, f.rbf_risk_score, f.knn_density, f.eccentricity])
        .collect()
}

/// Compute the cross-product matrix C = MᵀM (n_features × n_features).
fn cross_product(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_rows = m.len();
    if n_rows == 0 {
        return vec![];
    }
    let n_cols = m[0].len();
    let mut c = vec![vec![0.0_f64; n_cols]; n_cols];
    for row in m {
        for i in 0..n_cols {
            for j in 0..n_cols {
                c[i][j] += row[i] * row[j];
            }
        }
    }
    c
}

/// Power iteration to find the dominant eigenpair of a symmetric matrix.
///
/// Returns `(eigenvalue, eigenvector)`.
fn power_iteration(a: &[Vec<f64>], max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = a.len();
    if n == 0 {
        return (0.0, vec![]);
    }
    // Initialise with all-ones, normalised.
    let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0_f64;

    for _ in 0..max_iter {
        // w = A v
        let mut w = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[i][j] * v[j];
            }
        }
        // Rayleigh quotient: λ = vᵀA v / vᵀv
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 {
            return (0.0, v);
        }
        let new_lambda = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum::<f64>() / v_norm_sq;

        // Normalise w
        let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if w_norm < 1e-30 {
            return (new_lambda, v);
        }
        for x in w.iter_mut() {
            *x /= w_norm;
        }

        // Check convergence
        if (new_lambda - lambda).abs() < tol {
            return (new_lambda, w);
        }
        lambda = new_lambda;
        v = w;
    }
    (lambda, v)
}

/// Deflate matrix A by subtracting λ v vᵀ (removes the dominant eigencomponent).
fn deflate(a: &[Vec<f64>], lambda: f64, v: &[f64]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut a2 = a.to_vec();
    for i in 0..n {
        for j in 0..n {
            a2[i][j] -= lambda * v[i] * v[j];
        }
    }
    a2
}

/// Full eigendecomposition of a symmetric positive semi-definite matrix via
/// power iteration + deflation.
fn eigen_decompose(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Vec::with_capacity(n);
    let mut residual = a.to_vec();

    for _ in 0..n {
        let (lambda, v) = power_iteration(&residual, 256, 1e-12);
        if lambda < 1e-15 {
            break; // remaining eigenvalues are effectively zero
        }
        eigenvalues.push(lambda);
        eigenvectors.push(v.clone());
        residual = deflate(&residual, lambda, &v);
    }
    (eigenvalues, eigenvectors)
}

/// Spectral embedding of the cell×feature matrix via SVD of M (via C = MᵀM).
///
/// # Algorithm
///
/// 1. Build M ∈ ℝ^(n×d) from [`CellFeatures`].
/// 2. Compute C = MᵀM (d×d, typically d = 5).
/// 3. Eigendecompose C → (λ_i, v_i).
/// 4. σ_i = √λ_i, u_i = (1/σ_i) M v_i.
/// 5. Effective rank = min k such that ∑₁ᵏ σ_i² ≥ 0.90 ∑₁ᵈ σ_i².
pub fn spectral_decompose(features: &[CellFeatures]) -> SpectralEmbedding {
    let m = build_incidence_matrix(features);
    let n_cells = m.len();
    if n_cells == 0 {
        return SpectralEmbedding {
            singular_values: vec![],
            left_vectors: vec![],
            right_vectors: vec![],
            effective_rank_90: 0,
            frobenius_energy_ratio: 0.0,
        };
    }

    let c = cross_product(&m);
    let (eigenvalues, right_vecs) = eigen_decompose(&c);

    // Singular values = sqrt(eigenvalues)
    let singular_values: Vec<f64> = eigenvalues.iter().map(|&l| l.sqrt()).collect();
    let rank = singular_values.len();

    // Total Frobenius energy: sum of squared singular values = sum of eigenvalues
    let total_energy: f64 = eigenvalues.iter().sum();

    // Effective rank at 90% variance
    let mut cumulative = 0.0_f64;
    let mut effective_rank_90 = rank;
    for (k, &sv) in singular_values.iter().enumerate() {
        cumulative += sv * sv;
        if cumulative >= 0.90 * total_energy {
            effective_rank_90 = k + 1;
            break;
        }
    }

    // Frobenius energy ratio for top-k (where k = effective_rank_90)
    let top_k_energy: f64 = singular_values
        .iter()
        .take(effective_rank_90)
        .map(|&s| s * s)
        .sum();
    let frobenius_energy_ratio = if total_energy > 1e-18 {
        top_k_energy / total_energy
    } else {
        1.0
    };

    // Left singular vectors: u_i = M v_i / σ_i
    let mut left_vectors: Vec<Vec<f64>> = vec![vec![0.0_f64; rank]; n_cells];
    for k in 0..rank {
        let sigma = singular_values[k];
        if sigma < 1e-18 {
            continue;
        }
        let inv_sigma = 1.0 / sigma;
        let v = &right_vecs[k];
        for i in 0..n_cells {
            let dot: f64 = m[i].iter().zip(v.iter()).map(|(mv, vv)| mv * vv).sum();
            left_vectors[i][k] = dot * inv_sigma;
        }
    }

    SpectralEmbedding {
        singular_values,
        left_vectors,
        right_vectors: right_vecs,
        effective_rank_90,
        frobenius_energy_ratio,
    }
}

/// Find cell pairs with near-zero dot product in the embedding space.
///
/// These pairs represent **decorrelated niches** — cells whose morphometric
/// profiles are nearly orthogonal, suggesting they belong to distinct
/// phenotypic compartments.
pub fn orthogonal_cell_pairs(
    embedding: &SpectralEmbedding,
    tol: f64,
) -> Vec<OrthogonalCellPair> {
    let n = embedding.left_vectors.len();
    if n < 2 {
        return vec![];
    }
    let mut pairs = Vec::new();
    // Use the first `effective_rank_90` dimensions for the dot product.
    let dim = embedding.effective_rank_90.min(embedding.left_vectors[0].len());

    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = (0..dim)
                .map(|k| embedding.left_vectors[i][k] * embedding.left_vectors[j][k])
                .sum();
            if dot.abs() <= tol {
                pairs.push(OrthogonalCellPair {
                    cell_a: i,
                    cell_b: j,
                    dot_product: dot,
                });
            }
        }
    }
    pairs
}

/// Assign cells to spectral clusters based on the sign pattern of their
/// top-2 embedding coordinates (quadrant-based clustering).
///
/// This is a simple, deterministic alternative to k-means that mirrors the
/// block-diagonal structure suggested by orthogonal cell pairs.
pub fn spectral_clusters(embedding: &SpectralEmbedding) -> Vec<usize> {
    let n = embedding.left_vectors.len();
    if n == 0 || embedding.left_vectors[0].len() < 2 {
        return vec![0; n];
    }
    embedding
        .left_vectors
        .iter()
        .map(|v| {
            let s0 = if v[0] >= 0.0 { 1 } else { 0 };
            let s1 = if v[1] >= 0.0 { 1 } else { 0 };
            s0 * 2 + s1 // 4 quadrants
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::CellFeatures;

    /// Build synthetic cells with two clearly separated populations.
    fn synthetic_two_clusters() -> Vec<CellFeatures> {
        let mut cells = Vec::new();
        // Cluster A: large, irregular cells
        for _ in 0..10 {
            cells.push(CellFeatures {
                area: 2.0,
                circularity: 0.35,
                rbf_risk_score: 0.6,
                knn_density: 0.5,
                eccentricity: 0.85,
            });
        }
        // Cluster B: small, round cells
        for _ in 0..10 {
            cells.push(CellFeatures {
                area: 0.8,
                circularity: 0.92,
                rbf_risk_score: 0.1,
                knn_density: 1.5,
                eccentricity: 0.05,
            });
        }
        cells
    }

    #[test]
    fn svd_detects_two_cluster_structure() {
        let cells = synthetic_two_clusters();
        let emb = spectral_decompose(&cells);

        // Should find at least 1 singular value (typically 2 for 2 clusters)
        assert!(!emb.singular_values.is_empty());
        assert!(emb.singular_values[0] > 0.0);

        // The effective rank should be small (structure is low-rank)
        assert!(emb.effective_rank_90 <= 3);

        // Frobenius energy ratio should be high for top components
        assert!(emb.frobenius_energy_ratio >= 0.85);

        // Left vectors should have the right shape
        assert_eq!(emb.left_vectors.len(), cells.len());
        assert_eq!(emb.left_vectors[0].len(), emb.singular_values.len());
    }

    #[test]
    fn orthogonal_pairs_from_antipodal_clusters() {
        let mut cells = Vec::new();
        // Two antipodal populations
        for _ in 0..5 {
            cells.push(CellFeatures {
                area: 2.0,
                circularity: 0.3,
                rbf_risk_score: 0.7,
                knn_density: 0.4,
                eccentricity: 0.9,
            });
        }
        for _ in 0..5 {
            cells.push(CellFeatures {
                area: 0.5,
                circularity: 0.9,
                rbf_risk_score: 0.05,
                knn_density: 2.0,
                eccentricity: 0.02,
            });
        }

        let emb = spectral_decompose(&cells);
        let pairs = orthogonal_cell_pairs(&emb, 0.1);

        // Should find some orthogonal pairs between the two clusters
        assert!(!pairs.is_empty(), "expected orthogonal pairs between clusters");
    }

    #[test]
    fn empty_input_produces_empty_embedding() {
        let emb = spectral_decompose(&[]);
        assert!(emb.singular_values.is_empty());
        assert!(emb.left_vectors.is_empty());
        assert_eq!(emb.effective_rank_90, 0);
    }

    #[test]
    fn spectral_clusters_partition_cells() {
        let cells = synthetic_two_clusters();
        let emb = spectral_decompose(&cells);
        let clusters = spectral_clusters(&emb);

        assert_eq!(clusters.len(), cells.len());
        // Clusters should be in 0..4 range (quadrant-based)
        assert!(clusters.iter().all(|&c| c < 4));

        // Cells from the same input cluster should generally map to
        // the same spectral quadrant (within each 10-cell block).
        let c0 = clusters[0];
        let all_same_in_block: bool = clusters[0..10].iter().all(|&c| c == c0);
        assert!(all_same_in_block, "first 10 cells should share spectral quadrant");
    }
}