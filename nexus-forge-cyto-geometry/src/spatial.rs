//! Phase-1/5 spatial analysis: morphometrics (area, perimeter, circularity, eccentricity), tumor-boundary distance,
//! centroids, kNN spacing, RBF field.
//!
//! **Why**: Deterministic pathology uses geometry-only cues—no imaging—so scalar shape descriptors feed Phase-2
//! tabular models alongside neighborhood graphs, interface topology, and smooth microenvironment fields.

/// Discrete clinical label attached after κ (**why**: optional metadata for downstream graph analytics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClinicalStatus {
    Normal,
    Malignant,
}

/// One cell’s planar embedding: morphometrics + graph/RBF quantities (**why**: strict `f64` for distances and ISO-style shape metrics).
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialNode {
    pub cell_id: u32,
    pub centroid_x: f64,
    pub centroid_y: f64,
    /// Absolute shoelace (Green’s) area in coordinate units².
    pub area: f64,
    /// Closed-ring perimeter: sum of Euclidean edge lengths.
    pub perimeter: f64,
    /// **4πA / P²** — disk ⇒ 1; irregular / star-like ⇒ toward 0 (**why**: standard compactness for Phase-2 tabular features).
    pub circularity: f64,
    /// Vertex-cloud eccentricity \( \sqrt{1 - \lambda_2/\lambda_1} \) from **population** \(2\times2\) covariance (**why**: SOTA elongation proxy).
    pub eccentricity: f64,
    /// Solidity = area / convex-hull area \(\in[0,1]\); concave or spiky membranes → lower (**why**: membrane irregularity).
    pub solidity: f64,
    /// Convexity = hull perimeter / perimeter \(\in(0,1]\); irregular membrane → lower (**why**: boundary roughness).
    pub convexity: f64,
    /// Aspect ratio \(\lambda_1 / \lambda_2\) of the vertex covariance (\(\ge 1\)).
    pub aspect_ratio: f64,
    /// Seven Hu shape invariants (log-scaled), rotation/scale-invariant shape descriptors.
    pub hu_moments: [f64; 7],
    /// Shortest distance from **centroid** to convex hull of **classifier-malignant** centroids; **`0`** if no malignant seeds (**why**: interface topology).
    pub distance_to_tumor_edge: f64,
    pub clinical_status: Option<ClinicalStatus>,
}

/// Absolute polygon area via shoelace / Green’s theorem (**why**: matches deterministic `(32,2)` boundary convention).
pub fn polygon_area_shoelace_abs(vertices: &[[f64; 2]]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        acc += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1];
    }
    0.5 * acc.abs()
}

/// Sum of Euclidean lengths along consecutive vertices, closing **`v[n-1]` → `v[0]`**.
pub fn polygon_perimeter(vertices: &[[f64; 2]]) -> f64 {
    let n = vertices.len();
    if n < 2 {
        return 0.0;
    }
    let mut p = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = vertices[j][0] - vertices[i][0];
        let dy = vertices[j][1] - vertices[i][1];
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

/// **C = (4π A) / P²** with safe denominator (**why**: disk approaches **1.0**; spiky shapes trend toward **0.0**).
pub fn polygon_circularity(area: f64, perimeter: f64) -> f64 {
    const P_EPS: f64 = 1e-18;
    if perimeter <= P_EPS || !area.is_finite() || !perimeter.is_finite() || area < 0.0 {
        return 0.0;
    }
    let c = (4.0 * std::f64::consts::PI * area) / (perimeter * perimeter);
    if !c.is_finite() {
        return 0.0;
    }
    c.min(1.0)
}

/// Population \(2\times2\) covariance eigenvalues \((\lambda_1 \ge \lambda_2)\) of the boundary samples.
///
/// Works with any number of ring points. Disk-like rings ⇒ \(\lambda_1 \approx \lambda_2\).
pub fn ring_covariance_eigenvalues(ring: &[[f32; 2]]) -> (f64, f64) {
    let n = ring.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let nf = n as f64;
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    for p in ring {
        mx += p[0] as f64;
        my += p[1] as f64;
    }
    mx /= nf;
    my /= nf;

    let mut cxx = 0.0_f64;
    let mut cyy = 0.0_f64;
    let mut cxy = 0.0_f64;
    for p in ring {
        let dx = p[0] as f64 - mx;
        let dy = p[1] as f64 - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    cxx /= nf;
    cyy /= nf;
    cxy /= nf;

    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = trace * trace - 4.0 * det;
    if !(disc.is_finite() && disc >= 0.0) {
        return (0.0, 0.0);
    }
    let root = disc.sqrt();
    (0.5 * (trace + root), 0.5 * (trace - root))
}

/// Elongation from **population** covariance of the boundary samples: \(e=\sqrt{\max(0,\,1-\lambda_2/\lambda_1)}\).
///
/// Disk-like rings ⇒ **`0`**; line-like spreads ⇒ toward **`1`**.
pub fn polygon_eccentricity(ring: &[[f32; 2]]) -> f64 {
    let (lambda1, lambda2) = ring_covariance_eigenvalues(ring);
    const EV_EPS: f64 = 1e-18;
    if lambda1 <= EV_EPS {
        return 0.0;
    }
    let ratio = (lambda2 / lambda1).clamp(0.0, 1.0);
    let inner = 1.0 - ratio;
    if inner <= 0.0 {
        return 0.0;
    }
    inner.sqrt()
}

/// Elongation ratio \(\lambda_1/\lambda_2\) of the vertex covariance (\(\ge 1\)).
pub fn polygon_aspect_ratio(ring: &[[f32; 2]]) -> f64 {
    let (lambda1, lambda2) = ring_covariance_eigenvalues(ring);
    const EV_EPS: f64 = 1e-18;
    if lambda2 <= EV_EPS || !lambda1.is_finite() {
        return 1.0;
    }
    (lambda1 / lambda2).max(1.0)
}

/// Seven Hu shape invariants (log-scaled for stability) from the boundary samples.
///
/// Computed from scale-invariant central moments up to 3rd order. These are rotation,
/// translation and scale invariant shape descriptors (analogous to `cv2.HuMoments`).
pub fn hu_moments_from_ring(ring: &[[f32; 2]]) -> [f64; 7] {
    let n = ring.len();
    if n < 3 {
        return [0.0; 7];
    }
    let nf = n as f64;
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    for p in ring {
        mx += p[0] as f64;
        my += p[1] as f64;
    }
    mx /= nf;
    my /= nf;

    let mut m11 = 0.0_f64;
    let mut m20 = 0.0_f64;
    let mut m02 = 0.0_f64;
    let mut m12 = 0.0_f64;
    let mut m21 = 0.0_f64;
    let mut m30 = 0.0_f64;
    let mut m03 = 0.0_f64;
    for p in ring {
        let x = p[0] as f64 - mx;
        let y = p[1] as f64 - my;
        m11 += x * y;
        m20 += x * x;
        m02 += y * y;
        m12 += x * y * y;
        m21 += x * x * y;
        m30 += x * x * x;
        m03 += y * y * y;
    }

    // Scale-invariant normalized central moments: eta = mu / mu00^(1 + (p+q)/2).
    let mu00 = nf;
    let n20 = m20 / (mu00 * mu00);
    let n02 = m02 / (mu00 * mu00);
    let n11 = m11 / (mu00 * mu00);
    let d3 = mu00 * mu00 * mu00.sqrt(); // mu00^2.5
    let n30 = m30 / d3;
    let n03 = m03 / d3;
    let n12 = m12 / d3;
    let n21 = m21 / d3;

    let h1 = n20 + n02;
    let h2 = (n20 - n02) * (n20 - n02) + 4.0 * n11 * n11;
    let h3 = (n30 - 3.0 * n12) * (n30 - 3.0 * n12) + (3.0 * n21 - n03) * (3.0 * n21 - n03);
    let h4 = (n30 + n12) * (n30 + n12) + (n21 + n03) * (n21 + n03);
    let h5 = (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12) * (n30 + n12) - 3.0 * (n21 + n03) * (n21 + n03))
        + (3.0 * n21 - n03) * (n21 + n03) * (3.0 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));
    let h6 = (n20 - n02) * ((n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03))
        + 4.0 * n11 * (n30 + n12) * (n21 + n03);
    let h7 = (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12) * (n30 + n12) - 3.0 * (n21 + n03) * (n21 + n03))
        - (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));

    let log_scale = |v: f64| -> f64 {
        if v == 0.0 {
            0.0
        } else {
            -v.abs().log10()
        }
    };
    [log_scale(h1), log_scale(h2), log_scale(h3), log_scale(h4), log_scale(h5), log_scale(h6), log_scale(h7)]
}

/// Andrew monotone-chain convex hull, CCW, no duplicate closing vertex (**why**: deterministic tumor-cluster boundary).
pub fn calculate_convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let n = points.len();
    if n <= 1 {
        return points.to_vec();
    }

    #[inline]
    fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    }

    let mut pts: Vec<[f64; 2]> = points.to_vec();
    pts.sort_by(|a, b| {
        // NaN-safe total ordering: NaN sorts *after* all finite values so that
        // NaN coordinates never silently compare as Equal to real numbers.
        fn nan_safe_cmp(a: f64, b: f64) -> std::cmp::Ordering {
            match a.partial_cmp(&b) {
                Some(ord) => ord,
                None => {
                    // At least one is NaN — push NaN to the end.
                    if a.is_nan() && b.is_nan() {
                        std::cmp::Ordering::Equal
                    } else if a.is_nan() {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    }
                }
            }
        }
        nan_safe_cmp(a[0], b[0]).then_with(|| nan_safe_cmp(a[1], b[1]))
    });

    let mut lower: Vec<[f64; 2]> = Vec::with_capacity(n / 2 + 1);
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<[f64; 2]> = Vec::with_capacity(n / 2 + 1);
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[inline]
fn dist_point_to_segment_sq(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let vx = b[0] - a[0];
    let vy = b[1] - a[1];
    let wx = p[0] - a[0];
    let wy = p[1] - a[1];
    let vv = vx * vx + vy * vy;
    if vv <= 1e-36 {
        return wx * wx + wy * wy;
    }
    let t = (wx * vx + wy * vy) / vv;
    let t = t.clamp(0.0, 1.0);
    let cx = a[0] + t * vx;
    let cy = a[1] + t * vy;
    let dx = p[0] - cx;
    let dy = p[1] - cy;
    dx * dx + dy * dy
}

/// Minimum Euclidean distance from **`point`** to **closed** polygon **`hull`** edges (**why**: signed depth omitted — pure boundary proximity).
///
/// When **`hull`** is empty (no malignant seeds), returns **`f64::INFINITY`** (**why**: a point cannot be
/// "on the boundary" of a tumor cluster that does not exist — infinity signals "infinitely far" and is
/// handled gracefully by downstream RBF/clustering code, unlike the previous `0.0` which was
/// indistinguishable from "exactly on the tumor edge").
pub fn distance_to_polygon(point: [f64; 2], hull: &[[f64; 2]]) -> f64 {
    let n = hull.len();
    match n {
        0 => f64::INFINITY,
        1 => {
            let dx = point[0] - hull[0][0];
            let dy = point[1] - hull[0][1];
            (dx * dx + dy * dy).sqrt()
        }
        2 => dist_point_to_segment_sq(point, hull[0], hull[1]).sqrt(),
        _ => {
            let mut best_sq = f64::INFINITY;
            for i in 0..n {
                let a = hull[i];
                let b = hull[(i + 1) % n];
                let d2 = dist_point_to_segment_sq(point, a, b);
                if d2 < best_sq {
                    best_sq = d2;
                }
            }
            best_sq.sqrt()
        }
    }
}

/// Shoelace centroid of a closed polygon sampled by **`vertices`** (**why**: center of mass for uniform boundary mass).
///
/// Returns **`None`** only when the polygon area collapses below **`area_eps`** (degenerate ring); callers may fall back.
pub fn polygon_centroid_xy(vertices: &[[f64; 2]]) -> Option<(f64, f64)> {
    let n = vertices.len();
    if n < 3 {
        let sx: f64 = vertices.iter().map(|p| p[0]).sum();
        let sy: f64 = vertices.iter().map(|p| p[1]).sum();
        let nf = n as f64;
        return Some((sx / nf.max(1.0), sy / nf.max(1.0)));
    }

    let mut a_twice = 0.0_f64;
    let mut cx_acc = 0.0_f64;
    let mut cy_acc = 0.0_f64;

    for i in 0..n {
        let j = (i + 1) % n;
        let xi = vertices[i][0];
        let yi = vertices[i][1];
        let xj = vertices[j][0];
        let yj = vertices[j][1];
        let cross = xi * yj - xj * yi;
        a_twice += cross;
        cx_acc += (xi + xj) * cross;
        cy_acc += (yi + yj) * cross;
    }

    let area = 0.5 * a_twice;
    const AREA_EPS: f64 = 1e-18;
    if area.abs() < AREA_EPS {
        let sx: f64 = vertices.iter().map(|p| p[0]).sum();
        let sy: f64 = vertices.iter().map(|p| p[1]).sum();
        let nf = n as f64;
        return Some((sx / nf, sy / nf));
    }

    let factor = 1.0 / (6.0 * area);
    Some((cx_acc * factor, cy_acc * factor))
}

/// Converts gold rings into [`SpatialNode`]s using [`polygon_centroid_xy`].
/// Works with any ring size (was fixed at 32; now generic).
///
/// **`clinical`** must align by index with **`rings`** when provided (**why**: κ labels attach after geometry exists).
pub fn spatial_nodes_from_rings(
    rings: &[Vec<[f32; 2]>],
    clinical: Option<&[Option<ClinicalStatus>]>,
) -> Vec<SpatialNode> {
    rings
        .iter()
        .enumerate()
        .map(|(idx, ring)| {
            let verts: Vec<[f64; 2]> = ring.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
            let (cx, cy) = polygon_centroid_xy(&verts).unwrap_or((0.0, 0.0));
            let area = polygon_area_shoelace_abs(&verts);
            let perimeter = polygon_perimeter(&verts);
            let circularity = polygon_circularity(area, perimeter);
            let eccentricity = polygon_eccentricity(ring);
            let aspect_ratio = polygon_aspect_ratio(ring);
            let hu_moments = hu_moments_from_ring(ring);
            let hull = calculate_convex_hull(&verts);
            let hull_area = polygon_area_shoelace_abs(&hull);
            let hull_perimeter = polygon_perimeter(&hull);
            let solidity = if hull_area > 1e-18 { (area / hull_area).clamp(0.0, 1.0) } else { 0.0 };
            let convexity = if perimeter > 1e-18 { (hull_perimeter / perimeter).min(1.0) } else { 0.0 };
            let clinical_status = clinical.and_then(|c| c.get(idx).copied().flatten());
            SpatialNode {
                cell_id: idx as u32,
                centroid_x: cx,
                centroid_y: cy,
                area,
                perimeter,
                circularity,
                eccentricity,
                solidity,
                convexity,
                aspect_ratio,
                hu_moments,
                distance_to_tumor_edge: 0.0,
                clinical_status,
            }
        })
        .collect()
}

#[inline]
fn euclidean_sq(a: &SpatialNode, b: &SpatialNode) -> f64 {
    let dx = a.centroid_x - b.centroid_x;
    let dy = a.centroid_y - b.centroid_y;
    dx * dx + dy * dy
}

/// Mean Euclidean distance from each cell to its **`k`** nearest other cells (**why**: smaller values ⇒ tighter packing).
///
/// For **`n ≤ 1`**, returns **`0.0`**. Effective neighbor count is **`min(k, n - 1)`** when **`n ≥ 2`**.
pub fn calculate_knn_density(nodes: &[SpatialNode], k: usize) -> Vec<f64> {
    let n = nodes.len();
    let mut out = vec![0.0_f64; n];
    if n <= 1 || k == 0 {
        return out;
    }

    let k_eff = k.min(n - 1).max(1);

    for i in 0..n {
        let mut dists: Vec<f64> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = euclidean_sq(&nodes[i], &nodes[j]).sqrt();
            // Skip NaN/inf distances so they don't corrupt the kNN average.
            if d.is_finite() {
                dists.push(d);
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| {
            // NaN-safe: total ordering pushing NaN to the end.
            if a.is_nan() && b.is_nan() {
                std::cmp::Ordering::Equal
            } else if a.is_nan() {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }));
        let take = k_eff.min(dists.len());
        if take == 0 {
            out[i] = 0.0;
        } else {
            let slice = &dists[..take];
            out[i] = slice.iter().sum::<f64>() / slice.len() as f64;
        }
    }

    out
}

/// Gaussian RBF aggregate field at **`target`** over all **other** nodes (**why**: smooth “microenvironment pressure” proxy).
///
/// \[
/// F(x) = \sum_{j \neq i} \exp\!\left(-\frac{d_{ij}^2}{2\sigma^2}\right)
/// \]
pub fn calculate_rbf_field(target: &SpatialNode, all_nodes: &[SpatialNode], sigma: f64) -> f64 {
    let sigma = sigma.max(1e-12);
    let denom = 2.0 * sigma * sigma;
    let mut acc = 0.0_f64;
    for o in all_nodes {
        if o.cell_id == target.cell_id {
            continue;
        }
        let d2 = euclidean_sq(target, o);
        acc += (-d2 / denom).exp();
    }
    acc
}

/// Default Gaussian width from median positive kNN spacing (**why**: scale-adaptive σ without image statistics).
pub fn default_rbf_sigma(knn_mean_distances: &[f64]) -> f64 {
    let mut v: Vec<f64> = knn_mean_distances
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| {
        // NaN-safe: filtered above, but defensive for any residual NaN.
        if a.is_nan() && b.is_nan() {
            std::cmp::Ordering::Equal
        } else if a.is_nan() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }));
    let mid = v.len() / 2;
    let med = if v.len() % 2 == 1 {
        v[mid]
    } else {
        0.5 * (v[mid - 1] + v[mid])
    };
    med.max(1e-9)
}

/// Recommended k for MoNuSeg-scale slides (~300 cells) (**why**: stable local density without huge neighborhoods).
pub const DEFAULT_KNN_K: usize = 8;

// ---------------------------------------------------------------------------
// Mapping #2: Density ratio graph features
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlignmentClass { PhiLike, PiLike, Balanced, None }

#[derive(Clone, Debug)]
pub struct DensityRatioEdge {
    pub source: usize, pub target: usize,
    pub ratio: f64, pub log_ratio: f64,
    pub alignment_class: AlignmentClass,
}

pub fn build_knn_graph(nodes: &[SpatialNode], k: usize) -> Vec<(usize, usize)> {
    let n = nodes.len(); if n <= 1 || k == 0 { return vec![]; }
    let k_eff = k.min(n - 1).max(1);
    let mut edges = Vec::with_capacity(n * k_eff);
    for i in 0..n {
        let mut dists: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
        for j in 0..n { if i == j { continue; }
            let dx = nodes[i].centroid_x - nodes[j].centroid_x;
            let dy = nodes[i].centroid_y - nodes[j].centroid_y;
            let d2 = dx * dx + dy * dy;
            if d2.is_finite() { dists.push((d2, j)); }
        }
        dists.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for &(_, j) in dists.iter().take(k_eff) { edges.push((i, j)); }
    }
    edges
}

pub fn compute_density_ratios(nodes: &[SpatialNode], knn_graph: &[(usize, usize)], density_fn: fn(&SpatialNode) -> f64) -> Vec<DensityRatioEdge> {
    const PHI: f64 = 1.618033988749895; const PI: f64 = 3.141592653589793;
    const PHI_TOL: f64 = 0.05; const PI_TOL: f64 = 0.05; const BALANCE_TOL: f64 = 0.02;
    knn_graph.iter().filter_map(|&(src, tgt)| {
        if src >= nodes.len() || tgt >= nodes.len() { return None; }
        let rho_src = density_fn(&nodes[src]); let rho_tgt = density_fn(&nodes[tgt]);
        if rho_tgt.abs() < 1e-18 { return None; }
        let ratio = rho_src / rho_tgt;
        if !ratio.is_finite() || ratio <= 0.0 { return None; }
        let log_ratio = ratio.ln();
        let alignment = if (ratio - 1.0).abs() < BALANCE_TOL { AlignmentClass::Balanced }
        else if (ratio - PHI).abs() < PHI_TOL || (ratio / PHI - 1.0).abs() < PHI_TOL { AlignmentClass::PhiLike }
        else if (ratio - PI).abs() < PI_TOL || (ratio / PI - 1.0).abs() < PI_TOL { AlignmentClass::PiLike }
        else { AlignmentClass::None };
        Some(DensityRatioEdge { source: src, target: tgt, ratio, log_ratio, alignment_class: alignment })
    }).collect()
}

#[derive(Clone, Debug)]
pub struct DensityRatioStats {
    pub n_edges: usize, pub mean_ratio: f64, pub std_ratio: f64,
    pub phi_like_count: usize, pub pi_like_count: usize, pub balanced_count: usize,
    pub alignment_fraction: f64,
}

pub fn density_ratio_summary(edges: &[DensityRatioEdge]) -> DensityRatioStats {
    let n = edges.len(); if n == 0 { return DensityRatioStats { n_edges:0,mean_ratio:0.0,std_ratio:0.0,phi_like_count:0,pi_like_count:0,balanced_count:0,alignment_fraction:0.0 }; }
    let mean = edges.iter().map(|e|e.ratio).sum::<f64>() / n as f64;
    let var = edges.iter().map(|e|{let d=e.ratio-mean;d*d}).sum::<f64>() / n as f64;
    let phi = edges.iter().filter(|e|e.alignment_class==AlignmentClass::PhiLike).count();
    let pi = edges.iter().filter(|e|e.alignment_class==AlignmentClass::PiLike).count();
    let bal = edges.iter().filter(|e|e.alignment_class==AlignmentClass::Balanced).count();
    DensityRatioStats { n_edges:n, mean_ratio:mean, std_ratio:var.sqrt(), phi_like_count:phi, pi_like_count:pi, balanced_count:bal, alignment_fraction:(phi+pi+bal) as f64 / n as f64 }
}

pub fn default_cell_density(node: &SpatialNode) -> f64 { node.area.max(1e-12) }

// ---------------------------------------------------------------------------
// Mapping #5: Scalable blocked RBF
// ---------------------------------------------------------------------------

pub struct RBFConfig { pub sigma: f64, pub n_latent: usize, pub block_size: usize }
impl Default for RBFConfig { fn default() -> Self { Self { sigma:1.0, n_latent:3072, block_size:256 } } }

pub fn calculate_rbf_field_blocked(target: &SpatialNode, all_nodes: &[SpatialNode], config: &RBFConfig) -> f64 {
    let sigma = config.sigma.max(1e-12); let denom = 2.0*sigma*sigma;
    let n = all_nodes.len().min(config.n_latent); let block = config.block_size.max(1);
    let mut acc = 0.0_f64;
    for chunk in all_nodes[..n].chunks(block) {
        let mut bs = 0.0_f64;
        for o in chunk { if o.cell_id == target.cell_id { continue; }
            let dx = target.centroid_x - o.centroid_x; let dy = target.centroid_y - o.centroid_y;
            bs += (-(dx*dx+dy*dy)/denom).exp(); }
        acc += bs;
    }
    acc
}

pub fn rbf_tissue_surface(nodes: &[SpatialNode], bounds: (f64,f64,f64,f64), resolution: usize, config: &RBFConfig) -> Vec<Vec<f64>> {
    let (x_min,x_max,y_min,y_max) = bounds; let dx = (x_max-x_min)/resolution as f64; let dy = (y_max-y_min)/resolution as f64;
    let mut surface = vec![vec![0.0_f64; resolution]; resolution];
    for i in 0..resolution { let x = x_min + (i as f64+0.5)*dx;
        for j in 0..resolution { let y = y_min + (j as f64+0.5)*dy;
            let target = SpatialNode { cell_id:u32::MAX, centroid_x:x, centroid_y:y, area:0.0, perimeter:0.0, circularity:0.0, eccentricity:0.0, solidity:0.0, convexity:0.0, aspect_ratio:1.0, hu_moments:[0.0; 7], distance_to_tumor_edge:0.0, clinical_status:None };
            surface[i][j] = calculate_rbf_field_blocked(&target, nodes, config);
        }
    }
    surface
}

// ---------------------------------------------------------------------------
// Mapping #8: Harmonic gradients
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HarmonicGradient { pub amplitude: f64, pub frequency: f64, pub phase: f64, pub exponential_decay: f64 }

pub fn fit_harmonic_gradient(positions: &[[f64;2]], values: &[f64], axis: &[f64;2]) -> Option<HarmonicGradient> {
    let n = positions.len(); if n < 4 || values.len() != n { return None; }
    let an = (axis[0]*axis[0]+axis[1]*axis[1]).sqrt(); if an < 1e-18 { return None; }
    let ux = axis[0]/an; let uy = axis[1]/an;
    let dists: Vec<f64> = positions.iter().map(|p| p[0]*ux + p[1]*uy).collect();
    let d_min = dists.iter().cloned().fold(f64::INFINITY, f64::min);
    let ds: Vec<f64> = dists.iter().map(|&d| d-d_min).collect();
    let pp: Vec<(f64,f64)> = ds.iter().zip(values.iter()).filter(|(_,&v)| v>1e-18 && v.is_finite()).map(|(&d,&v)| (d, v.ln())).collect();
    if pp.len() < 3 { return None; }
    let md = pp.iter().map(|(d,_)|d).sum::<f64>()/pp.len() as f64;
    let ml = pp.iter().map(|(_,l)|l).sum::<f64>()/pp.len() as f64;
    let mut num=0.0_f64; let mut den=0.0_f64;
    for (d,l) in &pp { let dd = d-md; num += dd*(l-ml); den += dd*dd; }
    let lambda = if den>1e-18 { -num/den } else { 0.0 };
    let amplitude = (ml + lambda*md).exp().max(1e-18);
    let sc = pp.windows(2).filter(|w| (w[0].1 - ml + lambda*w[0].0).signum() != (w[1].1 - ml + lambda*w[1].0).signum()).count();
    let freq = sc as f64 / (2.0 * ds.last().copied().unwrap_or(1.0).max(1e-9));
    Some(HarmonicGradient { amplitude, frequency: freq.max(0.0), phase: 0.0, exponential_decay: lambda.max(0.0) })
}

pub fn path_integral_along_trajectory(path: &[[f64;2]], field: &dyn Fn([f64;2]) -> f64) -> f64 {
    if path.len() < 2 { return 0.0; }
    let mut integral = 0.0_f64;
    for w in path.windows(2) { let a=w[0]; let b=w[1]; let ds = ((b[0]-a[0]).powi(2)+(b[1]-a[1]).powi(2)).sqrt(); let mid=[(a[0]+b[0])*0.5,(a[1]+b[1])*0.5]; integral += field(mid)*ds; }
    integral
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_centroid_unit_square_corners() {
        let sq: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (cx, cy) = polygon_centroid_xy(&sq).unwrap();
        assert!((cx - 0.5).abs() < 1e-9 && (cy - 0.5).abs() < 1e-9);
    }

    #[test]
    fn knn_two_points_mean_distance_is_pair_distance() {
        let nodes = vec![
            SpatialNode {
                cell_id: 0,
                centroid_x: 0.0,
                centroid_y: 0.0,
                area: 0.0,
                perimeter: 0.0,
                circularity: 0.0,
                eccentricity: 0.0,
                solidity: 0.0,
                convexity: 0.0,
                aspect_ratio: 1.0,
                hu_moments: [0.0; 7],
                distance_to_tumor_edge: 0.0,
                clinical_status: None,
            },
            SpatialNode {
                cell_id: 1,
                centroid_x: 3.0,
                centroid_y: 4.0,
                area: 0.0,
                perimeter: 0.0,
                circularity: 0.0,
                eccentricity: 0.0,
                solidity: 0.0,
                convexity: 0.0,
                aspect_ratio: 1.0,
                hu_moments: [0.0; 7],
                distance_to_tumor_edge: 0.0,
                clinical_status: None,
            },
        ];
        let d = calculate_knn_density(&nodes, 8);
        assert!((d[0] - 5.0).abs() < 1e-9);
        assert!((d[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn rbf_two_cells_matches_closed_form() {
        let nodes = vec![
            SpatialNode {
                cell_id: 0,
                centroid_x: 0.0,
                centroid_y: 0.0,
                area: 0.0,
                perimeter: 0.0,
                circularity: 0.0,
                eccentricity: 0.0,
                solidity: 0.0,
                convexity: 0.0,
                aspect_ratio: 1.0,
                hu_moments: [0.0; 7],
                distance_to_tumor_edge: 0.0,
                clinical_status: None,
            },
            SpatialNode {
                cell_id: 1,
                centroid_x: 1.0,
                centroid_y: 0.0,
                area: 0.0,
                perimeter: 0.0,
                circularity: 0.0,
                eccentricity: 0.0,
                solidity: 0.0,
                convexity: 0.0,
                aspect_ratio: 1.0,
                hu_moments: [0.0; 7],
                distance_to_tumor_edge: 0.0,
                clinical_status: None,
            },
        ];
        let sigma = 1.0;
        let f0 = calculate_rbf_field(&nodes[0], &nodes, sigma);
        let expected = (-1.0_f64 / (2.0 * sigma * sigma)).exp();
        assert!((f0 - expected).abs() < 1e-12);
        let f1 = calculate_rbf_field(&nodes[1], &nodes, sigma);
        assert!((f1 - expected).abs() < 1e-12);
    }

    #[test]
    fn spatial_nodes_from_rings_preserves_ids() {
        let ring = [[1.0_f32, 1.0_f32]; 32];
        let nodes = spatial_nodes_from_rings(&[ring.to_vec()], None);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].cell_id, 0);
        assert_eq!(nodes[0].clinical_status, None);
        assert_eq!(nodes[0].area, 0.0);
        assert_eq!(nodes[0].perimeter, 0.0);
        assert_eq!(nodes[0].circularity, 0.0);
        assert_eq!(nodes[0].eccentricity, 0.0);
        assert_eq!(nodes[0].distance_to_tumor_edge, 0.0);
    }

    #[test]
    fn polygon_eccentricity_circle_ring_near_zero() {
        let mut ring = [[0.0_f32; 2]; 32];
        for (i, pt) in ring.iter_mut().enumerate() {
            let t = (i as f32) / 32.0 * std::f32::consts::TAU;
            *pt = [t.cos(), t.sin()];
        }
        let e = polygon_eccentricity(&ring);
        assert!(
            e < 0.08,
            "expected near-isotropic sampling → low eccentricity, got {e}"
        );
    }

    #[test]
    fn polygon_eccentricity_segment_cloud_is_high() {
        let mut ring = [[0.0_f32; 2]; 32];
        for (i, pt) in ring.iter_mut().enumerate() {
            let x = i as f32;
            *pt = [x, 0.02 * (i % 3) as f32];
        }
        let e = polygon_eccentricity(&ring);
        assert!(e > 0.75, "expected strongly elongated cloud, got {e}");
    }

    #[test]
    fn convex_hull_square_four_vertices() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
        let h = calculate_convex_hull(&pts);
        assert_eq!(h.len(), 4);
        let set: std::collections::HashSet<(i64, i64)> = h
            .iter()
            .map(|p| {
                (
                    (p[0] * 1000.0).round() as i64,
                    (p[1] * 1000.0).round() as i64,
                )
            })
            .collect();
        assert!(set.contains(&(0, 0)));
        assert!(set.contains(&(1000, 0)));
        assert!(set.contains(&(1000, 1000)));
        assert!(set.contains(&(0, 1000)));
    }

    #[test]
    fn distance_to_polygon_center_of_square() {
        let hull = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let d = distance_to_polygon([1.0, 1.0], &hull);
        assert!((d - 1.0).abs() < 1e-9);
    }

    #[test]
    fn unit_square_morphometrics_disk_compactness_reference() {
        let sq: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let a = polygon_area_shoelace_abs(&sq);
        let p = polygon_perimeter(&sq);
        let c = polygon_circularity(a, p);
        assert!((a - 1.0).abs() < 1e-12);
        assert!((p - 4.0).abs() < 1e-12);
        let expected = std::f64::consts::PI / 4.0;
        assert!((c - expected).abs() < 1e-12);
    }

    #[test]
    fn circularity_zero_when_perimeter_collapses() {
        assert_eq!(polygon_circularity(1.0, 0.0), 0.0);
        assert_eq!(polygon_circularity(1.0, 1e-30), 0.0);
    }
}
