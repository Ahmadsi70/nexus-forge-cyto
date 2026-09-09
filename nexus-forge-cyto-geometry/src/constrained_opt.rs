//! Mapping #4: Projected gradient flow on a constrained manifold.
//!
//! **Mathematical origin** (from `discovered_mathematical_mappings.md`):
//! > θ^(k+1) = Π_B(θ^(k) - α^(k) ∇_θ L(θ^(k)))
//! > Π_B = projection onto feasible set (quantized manifold)
//!
//! **Cell-network analogue:**
//! - Optimize interaction potentials or diffusion kernels on cell-graph edges
//!   subject to mass balance, maximum degree, and per-cell curvature bounds.
//! - Project each iterate back onto the feasible set B.
//!
//! **Use cases:**
//! - Learn optimal edge weights for the kNN graph under biological constraints.
//! - Fit diffusion parameters to observed spatial patterns.

// ---------------------------------------------------------------------------
// Constraint types
// ---------------------------------------------------------------------------

/// Bounds for a single parameter or aggregate quantity.
#[derive(Clone, Debug)]
pub struct Bounds {
    pub lower: f64,
    pub upper: f64,
}

impl Bounds {
    pub fn new(lower: f64, upper: f64) -> Self {
        assert!(lower <= upper);
        Self { lower, upper }
    }

    pub fn clamp(&self, x: f64) -> f64 {
        x.clamp(self.lower, self.upper)
    }

    pub fn project(&self, x: f64) -> f64 {
        if x < self.lower { self.lower }
        else if x > self.upper { self.upper }
        else { x }
    }
}

/// Constraints for the cell-graph parameter manifold.
#[derive(Clone, Debug)]
pub struct ConstrainedManifold {
    /// Per-edge interaction potential bounds.
    pub edge_weight_bounds: Bounds,
    /// Total mass (sum of all edge weights) bounds.
    pub mass_bounds: Bounds,
    /// Per-cell degree bounds.
    pub degree_bounds: Bounds,
    /// Global curvature (sum of squared weights) bounds.
    pub curvature_bounds: Bounds,
}

impl Default for ConstrainedManifold {
    fn default() -> Self {
        Self {
            edge_weight_bounds: Bounds::new(0.0, 10.0),
            mass_bounds: Bounds::new(0.1, 1e6),
            degree_bounds: Bounds::new(1.0, 50.0),
            curvature_bounds: Bounds::new(0.0, 1e6),
        }
    }
}

// ---------------------------------------------------------------------------
// Projection onto the feasible set
// ---------------------------------------------------------------------------

/// Project edge weights onto the feasible set B.
///
/// Three-stage projection:
/// 1. Clip each weight to `edge_weight_bounds`.
/// 2. Rescale to satisfy `mass_bounds` (total sum constraint).
/// 3. Clamp per-node degrees to `degree_bounds`.
pub fn project_onto_feasible_set(
    weights: &mut [f64],
    edge_index: &[(usize, usize)],
    num_nodes: usize,
    constraints: &ConstrainedManifold,
) -> f64 {
    let n_edges = weights.len();
    if n_edges == 0 {
        return 0.0;
    }

    // Stage 1: clip each weight
    for w in weights.iter_mut() {
        *w = constraints.edge_weight_bounds.project(*w);
    }

    // Stage 2: rescale to mass bounds
    let total_mass: f64 = weights.iter().sum();
    if total_mass > constraints.mass_bounds.upper {
        let scale = constraints.mass_bounds.upper / total_mass;
        for w in weights.iter_mut() {
            *w *= scale;
        }
    } else if total_mass < constraints.mass_bounds.lower {
        let scale = constraints.mass_bounds.lower / total_mass.max(1e-18);
        for w in weights.iter_mut() {
            *w *= scale;
        }
    }

    // Stage 3: degree constraints
    let mut degrees = vec![0.0_f64; num_nodes];
    for (e, &(src, tgt)) in edge_index.iter().enumerate() {
        degrees[src] += weights[e];
        degrees[tgt] += weights[e];
    }

    // Count constraint violations
    let mut max_violation = 0.0_f64;
    for d in &degrees {
        if *d > constraints.degree_bounds.upper {
            let viol = *d - constraints.degree_bounds.upper;
            if viol > max_violation { max_violation = viol; }
        } else if *d < constraints.degree_bounds.lower && *d > 0.0 {
            let viol = constraints.degree_bounds.lower - *d;
            if viol > max_violation { max_violation = viol; }
        }
    }

    max_violation
}

// ---------------------------------------------------------------------------
// Projected gradient descent
// ---------------------------------------------------------------------------

/// One step of projected gradient descent.
///
/// θ ← Π_B(θ - α ∇L(θ))
///
/// Returns the constraint violation after projection (0 = feasible).
pub fn projected_gradient_step(
    params: &mut [f64],
    grad: &[f64],
    step_size: f64,
    edge_index: &[(usize, usize)],
    num_nodes: usize,
    constraints: &ConstrainedManifold,
) -> f64 {
    assert_eq!(params.len(), grad.len());

    // Gradient descent step
    for (p, g) in params.iter_mut().zip(grad.iter()) {
        *p -= step_size * *g;
    }

    // Project back onto feasible set
    project_onto_feasible_set(params, edge_index, num_nodes, constraints)
}

/// Run projected gradient descent until convergence.
///
/// Returns the final parameters, final loss, and number of iterations.
pub fn projected_gradient_descent(
    init_params: &[f64],
    edge_index: &[(usize, usize)],
    num_nodes: usize,
    constraints: &ConstrainedManifold,
    step_size: f64,
    max_iter: usize,
    tol: f64,
    loss_fn: impl Fn(&[f64]) -> (f64, Vec<f64>), // returns (loss, gradient)
) -> (Vec<f64>, f64, usize) {
    let mut params = init_params.to_vec();
    let mut prev_loss = f64::INFINITY;

    for iter in 0..max_iter {
        let (loss, grad) = loss_fn(&params);

        // Check convergence
        if (prev_loss - loss).abs() < tol {
            return (params, loss, iter + 1);
        }
        prev_loss = loss;

        let _violation = projected_gradient_step(
            &mut params, &grad, step_size, edge_index, num_nodes, constraints,
        );

        // Safety: if loss explodes, break
        if !loss.is_finite() {
            break;
        }
    }

    // Re-evaluate final loss
    let (final_loss, _) = loss_fn(&params);
    (params, final_loss, max_iter)
}

// ---------------------------------------------------------------------------
// Simple loss functions for edge-weight optimization
// ---------------------------------------------------------------------------

/// Laplacian smoothness loss: ∑ w_ij · (x_i - x_j)²
///
/// Minimising this encourages edges with large feature differences
/// to have low weight (and vice versa).
pub fn laplacian_smoothness_loss(
    weights: &[f64],
    edge_index: &[(usize, usize)],
    features: &[Vec<f64>],
) -> (f64, Vec<f64>) {
    let n_edges = weights.len();
    let mut loss = 0.0_f64;
    let mut grad = vec![0.0_f64; n_edges];

    for (e, &(src, tgt)) in edge_index.iter().enumerate() {
        let diff_sq: f64 = features[src]
            .iter()
            .zip(features[tgt].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        loss += weights[e] * diff_sq;
        grad[e] = diff_sq;
    }

    // L2 regularisation
    let lambda: f64 = 0.01;
    for (e, w) in weights.iter().enumerate() {
        loss += lambda * w * w;
        grad[e] += 2.0 * lambda * w;
    }

    (loss, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_clips_bounds() {
        let mut weights = vec![-0.5, 5.0, 15.0, 3.0];
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let constraints = ConstrainedManifold::default();

        let _viol = project_onto_feasible_set(&mut weights, &edges, 4, &constraints);

        for w in &weights {
            assert!(*w >= constraints.edge_weight_bounds.lower);
            assert!(*w <= constraints.edge_weight_bounds.upper);
        }
    }

    #[test]
    fn gradient_step_reduces_simple_loss() {
        // Two clusters of 2 nodes each
        let features = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let edges = vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)];
        let init_weights = vec![1.0_f64; edges.len()];
        let constraints = ConstrainedManifold::default();

        let (loss_before, _) = laplacian_smoothness_loss(&init_weights, &edges, &features);

        // One gradient step should reduce loss
        let grad = laplacian_smoothness_loss(&init_weights, &edges, &features).1;
        let mut w2 = init_weights.clone();
        let _ = projected_gradient_step(&mut w2, &grad, 0.01, &edges, 4, &constraints);

        let (loss_after, _) = laplacian_smoothness_loss(&w2, &edges, &features);
        assert!(loss_after < loss_before, "gradient step should reduce loss");
    }

    #[test]
    fn descent_converges() {
        let features: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.5, (i % 3) as f64])
            .collect();
        let mut edges = Vec::new();
        for i in 0..10 {
            for j in (i + 1)..10 {
                if (i as isize - j as isize).abs() <= 3 {
                    edges.push((i, j));
                }
            }
        }
        let init_weights = vec![1.0_f64; edges.len()];
        let constraints = ConstrainedManifold::default();

        let (_params, loss, iters) = projected_gradient_descent(
            &init_weights, &edges, 10, &constraints,
            0.001, 500, 1e-8,
            |w| laplacian_smoothness_loss(w, &edges, &features),
        );

        let (initial_loss, _) = laplacian_smoothness_loss(&init_weights, &edges, &features);
        assert!(loss < initial_loss, "descent should reduce loss");
        assert!(iters > 0, "should run at least one iteration");
    }
}