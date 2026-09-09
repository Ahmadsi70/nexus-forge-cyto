# Nexus-Forge Cyto — Technical Architecture & Scope

**Document type:** Enterprise architecture & boundary specification  
**Authority:** Grounded in the current `nexus-forge-cyto` codebase (Rust library + Python `coord_adapter` + QuPath Groovy).  
**Audience:** Engineering, computational pathology, QA/regulatory stakeholders.

---

## Executive summary

**Nexus-Forge Cyto** is a **deterministic**, **geometry-first** cytology analytics engine: expert or dataset-derived **closed polygons** are normalized to fixed **`(32 × 2)`** vertex rings, enriched with **pure Rust** morphometrics and spatial topology metrics, scored by a **transparent weighted heuristic** (no trained neural model), augmented with **convex-hull interface distances**, and exported for **QuPath** measurement tables. Optional linkage to a **Modular Mojo** shared library (`math_core`) provides curvature κ; when that library is **not** linked, a **documented fallback** keeps the pipeline runnable with **κ = 0**.

---

## 1. The deterministic pipeline (step-by-step flow)

End-to-end enrichment for **segment JSON** (gold envelope) is orchestrated in **`src/geometric_orchestrator.rs`** — principally **`write_enriched_geometric_output`**. Batch directory processing is implemented in **`src/main.rs`** (**`nexus-forge-cyto-batch`** binary). Polygon ingestion from MoNuSeg-style sources is implemented in Python **`services/coord_adapter.py`**.

### Stage 1 — Data adaptation (polygon → fixed rings)

| Aspect | Implementation |
|--------|----------------|
| **In scope** | Parsing **MoNuSeg / ImageScope XML** (`Regions` → `Vertices`), **polygon JSON**, and **CSV vertex tables**; resampling each polygon to exactly **`TARGET_RING = 32`** points in **`(x, y)`**; emitting the segment envelope consumed by Rust (`cell_count`, `cells`, metadata). |
| **Primary artifacts** | **`services/coord_adapter.py`**: `parse_monuseg_xml`, `ring32_from_polygon_xy`, `segment_envelope_from_polygons`, `write_gold_segment_json`; helper **`tests/validation/gen_gold_from_xml.py`** for CLI-style XML → JSON. |
| **Rust contract** | **`write_enriched_geometric_output`** in **`geometric_orchestrator.rs`**: consumes gold segment JSON (`cell_count`, `cells` with **32** vertices per cell) and emits enriched morphometric + clinical JSON. |
| **Downstream** | JSON bytes deserialize → validated ring dimensions before any metric computation. |

---

### Stage 2 — Single-cell morphometrics (pure Rust)

| Metric | Method | Source module |
|--------|--------|----------------|
| **Area** | Shoelace / Green’s theorem on the **32-vertex ring** (absolute area). | **`spatial.rs`** — `polygon_area_shoelace_abs` |
| **Perimeter** | Sum of Euclidean edge lengths, **closed** ring. | **`polygon_perimeter`** |
| **Circularity** | \(C = 4\pi A / P^2\), clamped; safe denominator if \(P \approx 0\). | **`polygon_circularity`** |
| **Centroid** | Shoelace centroid (degenerate-area fallback to vertex mean). | **`polygon_centroid_xy`** |
| **Eccentricity** | Population covariance of the **32** boundary samples as a **2×2** matrix; eigenvalues \(\lambda_1 \ge \lambda_2\); \(e = \sqrt{\max(0,\,1-\lambda_2/\lambda_1)}\). | **`polygon_eccentricity`** |

**Structural type:** **`SpatialNode`** (`spatial.rs`) carries `area`, `perimeter`, `circularity`, `eccentricity`, centroid, and placeholders filled in later stages (`distance_to_tumor_edge`, optional label metadata).

**Parallel path (legacy gate):** **`geometric_orchestrator`** computes per-ring shoelace area in **`f32`** for a **peer median** rule (**1.5 × median area**) that can mark baseline **`is_malignant`** before tabular fusion; final **`clinical`** strings follow Stage 4 after classifier overwrite.

---

### Stage 3 — Microenvironment spatial topology (pure Rust)

| Metric | Definition | Source |
|--------|------------|--------|
| **kNN density** | Mean Euclidean distance from each cell centroid to its **`k`** nearest **other** cell centroids (**`DEFAULT_KNN_K`** in `spatial.rs`, MoNuSeg-scale default **8**). | **`calculate_knn_density`** |
| **RBF risk field** | Sum of Gaussian kernels \(\sum_{j \neq i} \exp(-d_{ij}^2 / (2\sigma^2))\) over other centroids; **σ** from **`default_rbf_sigma`** (median of positive kNN means). | **`calculate_rbf_field`**, **`default_rbf_sigma`** |

**Exported JSON:** Per-cell **`knn_density`** (raw mean distance) and **`rbf_risk_score`** (unnormalized field value). The tabular classifier consumes **slide-normalized** ratios (area / median area; kNN / median kNN) and **RBF normalized by `max(1, n_cells − 1)`** — see **`CellFeatures`** documentation in **`classifier.rs`**.

---

### Stage 4 — Explainable scoring matrix (pure Rust)

| Element | Specification |
|---------|----------------|
| **Inputs** | **`CellFeatures`**: normalized area ratio, circularity, normalized RBF, normalized kNN ratio, **eccentricity** \([0,1]\). |
| **Excluded from score** | **`distance_to_tumor_edge`** is **not** fed into **`malignancy_score`** (avoids circular dependence on hull built from post-classifier malignant labels). |
| **Fusion** | Weighted sum of five **clamped subscores** in **`[0,1]`**; weights **must sum to 1.0** (`debug_assert!` in **`malignancy_score`**). |
| **Decision** | **`malignancy_score > 0.65`** ⇒ **`ClinicalStatus::Malignant`**, else **`Normal`**. |

**Current weights** (`classifier.rs`):  

| Component | Weight | Role |
|-----------|--------|------|
| Area pleomorphism | **0.17** | Ramp above median after onset ratio |
| Irregular membrane (circularity) | **0.21** | Penalty below soft cap **0.75** |
| RBF microenvironment | **0.21** | Normalized field × gain, clamped |
| Packing / clustering (kNN ratio) | **0.17** | Tighter-than-median spacing |
| Eccentricity | **0.24** | Elongation biomarker |

**Output:** **`CytoCellResult::is_malignant`** updated and **`clinical`** JSON array (**`"Malignant"`** / **`"Normal"`**) aligned index-wise with **`cells`**.

---

### Stage 5 — Advanced spatial interface (pure Rust)

| Element | Implementation |
|---------|----------------|
| **Tumor boundary proxy** | Convex hull (**Andrew monotone chain**) over **centroids** of cells classified **malignant after Stage 4**. |
| **Per-cell distance** | Minimum Euclidean distance from **each** cell’s centroid to **closed polygon** hull edges (segment-wise); degenerate cases: **0** malignant seeds ⇒ empty hull ⇒ distance **0**; one seed ⇒ point distance; two seeds ⇒ segment distance. |
| **JSON** | **`spatial_features[].distance_to_tumor_edge`**, **`eccentricity`**, and all Stage 2–3 metrics per cell. |
| **Analysis block** | **`spatial_analysis`**: e.g. **`knn_k`**, **`rbf_sigma`**, **`knn_metric`** string. |

---

### Stage 6 — UI rendering (QuPath)

| Aspect | Implementation |
|--------|----------------|
| **Artifact** | **`scripts/qupath_enriched_visualizer.groovy`** (QuPath **v0.7+**). |
| **Input** | Enriched JSON path (Gson **`Map`** parse). |
| **Geometry** | One **`PathAnnotationObject`** per cell from **`cells[i]`** rings; **`PathClass`** coloring (**Nexus: Malignant** / **Nexus: Normal**). |
| **Measurements** | **`MeasurementList`**: **`Area_Math`**, **`Circularity`**, **`KNN_Density`**, **`RBF_Risk_Score`**, **`Perimeter_Math`**, **`Eccentricity`**, **`Tumor_Edge_Distance`** — enabling native sort/filter in QuPath. |
| **Hierarchy** | **`addObjects`** + **`fireHierarchyUpdate`**. |

---

### Graceful fallback — Mojo curvature (κ)

| Condition | Behavior |
|-----------|----------|
| **`invoke_mojo_curvature`** returns **`CytoErrMojoLibraryUnset`** (build without linked **`math_core`**, per **`mojo_bind.rs`** stub) | **Single** process-wide **`eprintln!`** warning; per ring **`Ok(CytoCellResult)`** with **`kappa = 0.0`**, **`is_malignant = 0`** *before* peer median gate; Stages 2–6 **unchanged**. |
| Other non-success Mojo codes | Still **`VisionClientError::Mojo`** — **not** silently masked when a real DLL is linked but rejects input. |

This preserves **auditability**: fallback is explicit and logged once; downstream biology remains **deterministic** from polygons + Rust math.

---

### Auxiliary delivery surfaces (in scope)

| Surface | Role |
|---------|------|
| **`nexus-forge-cyto-batch`** (`main.rs`) | Parallel (**Rayon**) batch over **`*.json`**; **`write_enriched_geometric_output`** per file; **`clap`** CLI. |
| **FFI / mmap pipeline** (`pipeline.rs`, `pipeline_parallel.rs`) | Separate **dense `(32, D)` tensor file** path for κ via Mojo when linked — distinct from segment JSON enrichment but shares **`CytoNeighborMatrixView`** / status taxonomy. |
| **Integration tests** | Library tests + **`tests/monuseg_tcga_fg_e2e.rs`** (XML → gold JSON → enriched JSON with optional Mojo stub). |

---

## 2. Deliberate exclusions (what is not used & why)

These boundaries align with **determinism**, **explainability**, **edge deployability**, and common **regulatory-preferred** positioning for fixed-algorithm clinical decision support (subject to independent validation — not a claim of clearance).

### 2.1 No deep learning / machine learning for classification

- **Excluded:** CNNs, GNNs, transformers, or any **trained** black-box scorer inside the Cyto enrichment path.
- **In use instead:** Fixed **`malignancy_score`** with **published constants** (weights, thresholds, gains) in **`classifier.rs`**; identical inputs ⇒ identical outputs.
- **Why:** Reduces **opacity** and **dataset-shift** risk from opaque embeddings; supports **FDA-style** rationale (“feature X exceeded weighted contribution Y”) without explaining millions of parameters.

### 2.2 No pixel-level computer vision

- **Excluded:** Raw WSI tiles, intensity histograms, texture (GLCM, etc.), entropy filters, semantic segmentation models on pixels.
- **In use instead:** **Expert-provided or dataset-exported polygons** only; all metrics are **pure geometry** and **centroid graphs** in coordinate space.
- **Why:** **Staining/scanner invariance** at the algorithm layer (subject to annotation coordinate consistency); predictable CPU/memory; avoids coupling to proprietary imaging stacks inside this crate.

### 2.3 No multi-omics / spatial transcriptomics

- **Excluded:** Gene expression, mutation panels, scRNA-seq fusion, spot-level Visium/MERFISH overlays.
- **In use instead:** **2D histology annotation coordinates** and derived spatial graphs only.
- **Why:** Scoped product boundary — Nexus-Forge Cyto as shipped here is a **spatial morphology + neighborhood** engine on **histology geometry**, not a genomics integrator.

### 2.4 No multiplex immunofluorescence / proteomics phenotyping

- **Excluded:** Multi-channel IF markers, antibody panels, spectral unmixing, protein co-expression clustering.
- **In use instead:** **Single effective phenotype** per cell from the **tabular heuristic** + optional κ when Mojo is present; QuPath classes are **binary-style** presentation labels for visualization.
- **Why:** Keeps the pipeline **marker-agnostic** and compatible with **standard brightfield/H&E-style** polygon exports that MoNuSeg-like adapters target.

---

## 3. Traceability matrix (stages → primary code locations)

| Stage | Key modules / files |
|-------|---------------------|
| 1 | `services/coord_adapter.py`, `tests/validation/gen_gold_from_xml.py` |
| 2 | `src/spatial.rs`, invoked from `src/geometric_orchestrator.rs` |
| 3 | `src/spatial.rs`, invoked from `src/geometric_orchestrator.rs` |
| 4 | `src/classifier.rs`, wired in `src/geometric_orchestrator.rs` |
| 5 | `src/spatial.rs` (`calculate_convex_hull`, `distance_to_polygon`), `src/geometric_orchestrator.rs` |
| 6 | `scripts/qupath_enriched_visualizer.groovy` |
| Fallback | `src/mojo_bind.rs` (stub), `src/geometric_orchestrator.rs` (Mojo κ fallback when library unset) |

---

## 4. Revision note

This document reflects the **repository state at authoring time**. Weight constants and thresholds live in **`classifier.rs`**; any change there should be mirrored in regulated **design documentation** and **release notes**.
