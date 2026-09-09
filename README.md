# 🧬 Nexus-Forge Cyto

**Explainable Geometry for Digital Pathology**

A multi-layered computational cytology platform for automating histopathology
analysis -- detecting and classifying malignant (cancerous) cells from
H&E-stained tissue slides using a unique combination of deep learning
computer vision, deterministic geometry, and spatial topology.

---

## What Makes Nexus-Forge Different

Most AI pathology tools are **black boxes** -- they tell you *where* cancer
is but not *why*. Nexus-Forge opens the box:

| Feature | Nexus-Forge | Paige / PathAI / Ibex |
|---------|------------|----------------------|
| **Explainability** | Published weights + math proofs | Billion-parameter black box |
| **Determinism** | Bit-for-bit identical outputs | GPU floating-point nondeterminism |
| **GPU Required** | No -- runs on CPU | Yes -- needs Azure/AWS GPU |
| **Regulatory path** | Auditable, published constants | Opaque embeddings |
| **Spatial topology** | kNN density + RBF risk field | Not computed |
| **Dual-path** | AI + Geometry | AI only |

---

## Architecture

The platform consists of 5 Rust crates forming a workspace, Python API
services, a React frontend, and an optional Mojo acceleration engine.

### Rust Crates (Cargo Workspace)

| Crate | Description |
|-------|-------------|
| **nexus-forge-cyto-geometry** | Shared deterministic geometry engine: shoelace area, perimeter, circularity, eccentricity, kNN density, RBF risk field, convex hull, explainable malignancy classifier with published weights. Pure Rust -- no AI, no FFI. |
| **nexus-forge-cyto** | Main pipeline: FFI surface (extern C), memory-mapped I/O, Mojo kappa curvature integration, batch processing with Rayon parallelism. Includes Python services: FastAPI server, Streamlit dashboard, SAM 3 segmentation. |
| **nexus-forge-cyto-ai** | AI-extended variant: adds HoVerNet ONNX inference, Cellpose segmentation, GBM fusion classifier (AUC 0.98), OpenSlide WSI support. |
| **nexus-forge-cyto-core** | Lightweight CLI binary wrapping the geometry engine. Two subcommands: Single and Batch. |
| **nexus-forge-cyto-api** | Axum-based HTTP REST API (Rust). Provides /api/health and /analyze-image endpoints. |

### Python Services

- **FastAPI Server** (port 8810): Dual ingress -- image (SAM 3 segmentation)
  and vector (XML/GeoJSON/JSON polygons). Enrichment via Rust bridge.
- **Streamlit Dashboard** (port 8501): 3-step interactive pipeline with
  Farsi/English UI, cancellable subprocess execution, artifact-gated
  navigation.
- **Coordinate Adapter**: Parses MoNuSeg XML, GeoJSON, JSON, and CSV
  polygon formats into unified 128-point ring representation.

### Frontend

- **nexus-viewer** (React 19 + TypeScript + Vite): Canvas-based polygon
  viewer with red (malignant) / blue (normal) color coding.
- **Legacy frontend**: Static HTML/CSS/JS dashboard with image upload.

## The Pipeline (6 Stages)

1. **Data Adaptation** -- XML/GeoJSON/JSON polygons normalized to fixed
   128-point rings
2. **Single-Cell Morphometrics** -- Shoelace area, perimeter, circularity,
   eccentricity (vertex covariance eigenvalues)
3. **Spatial Topology** -- kNN density (mean distance to K nearest
   neighbors), RBF risk field (Gaussian kernel sum over other centroids)
4. **Explainable Scoring** -- Weighted sum of 5 subscores (area
   pleomorphism 0.17, membrane irregularity 0.21, RBF microenvironment
   0.21, packing/clustering 0.17, eccentricity 0.24). Weights must sum to
   1.0 (runtime assertion). Threshold 0.35 -> Malignant.
5. **Tumor Interface** -- Convex hull (Andrew monotone chain) over
   malignant centroids, per-cell distance to tumor edge
6. **Visualization** -- QuPath Groovy script: red/amber/green annotations
   with measurement table (Area, Circularity, KNN Density, RBF Risk Score,
   Perimeter, Eccentricity, Tumor Edge Distance)

---

## Quickstart

### Prerequisites

- **Rust** >= 1.75 ([rustup.rs](https://rustup.rs))
- **Python** >= 3.10
- Optional: **Mojo** 0.26.1.0 (for kappa curvature acceleration)

### 5 Minutes to First Enriched Output

```bash
# 1. Clone
git clone https://github.com/clinicalguard/nexus-forge-cyto.git
cd cancer_project

# 2. Build Rust binaries (release mode)
cargo build --release

# 3. Run the pure-geometry pipeline on sample data
cargo run --bin nexus-core-cli -- \
  Single --input ./data/sample.json --output ./output.json

# 4. View results
cat ./output.json | jq '.clinical[:5]'
# -> ["Malignant", "Normal", "Normal", "Malignant", "Normal"]
```

### Topology Segmentation (Material/Void)

Segment binary topology (material/void) from physical field inputs using the
fine-tuned HoVerNet ONNX model. Works on CPU only — no GPU required.

```bash
# Show model info
python run_topology_inference.py --info

# Segment from an image (resized to 256x256 internally)
python run_topology_inference.py input.png -o output

# Segment from npz with 'image' key (256,256,3) uint8 HWC
python run_topology_inference.py input.npz -o output

# Quick self-test on synthetic pattern
python run_topology_inference.py --demo -o demo
```

Outputs: `output.npz` (foreground probability 164x164) + `output.png` (binary mask).

### TopoNet — Complementary Topology Tools

TopoNet is designed as a **complement** to existing structural optimization
software, not a replacement. It provides fast initial predictions that
accelerate the iterative solvers:

| Tool | Command | Purpose |
|------|---------|---------|
| **Interactive Previewer** | `streamlit run topology_demo.py` | Adjust load/volume-fraction sliders, see instant topology prediction. Ideal for education & exploration. |
| **Inference CLI** | `python run_topology_inference.py img.png -o out` | Batch/standalone prediction from images or npz. |
| **Abaqus Plugin** | `python topo_abaqus_plugin.py --demo` | Generates an Abaqus `.inp` with material/void element sets as a warm start. |

```bash
# Abaqus warm-start (demo mode)
python topo_abaqus_plugin.py -o initial_design.inp

# From displacement CSV (x,y,disp_x,disp_y)
python topo_abaqus_plugin.py --csv disps.csv --mesh-x 180 --mesh-y 60 -o initial.inp
```

### Docker (Full Stack)

```bash
cd nexus-forge-cyto
docker-compose up -d

# Services:
#   Streamlit Dashboard -> http://localhost:8501
#   FastAPI Server      -> http://localhost:8810/health
#   Axum API Server     -> http://localhost:8811/api/health
```

---

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `nexus-forge-cyto-geometry/` | Shared deterministic geometry engine (Rust library) |
| `nexus-forge-cyto/` | Main pipeline: FFI, mmap, Mojo kappa, batch CLI, Python services |
| `nexus-forge-cyto-ai/` | AI-extended: HoVerNet ONNX, Cellpose, SAM 3, GNN |
| `nexus-forge-cyto-core/` | Pure geometry CLI binary wrapper |
| `nexus-forge-cyto-api/` | Axum HTTP API server (port 8811) |
| `nexus-viewer/` | React frontend (canvas polygon visualization) |
| `models/` | Deployed models: HoVerNet ONNX, GBM classifier, fusion metadata |
| `training/` | Training scripts: HoVerNet fine-tuning, GBM retraining, ONNX export |
| `mojo_core/` | Mojo SIMD curvature engine (kappa_engine.mojo) |
| `data/` | Sample datasets and benchmark results |
| `docs/` | Architecture docs, integration strategy, API reference |

---

## Technologies

| Layer | Stack |
|-------|-------|
| **AI / ML** | HoVerNet (ONNX), Gradient Boosting (scikit-learn), PyTorch, ONNX Runtime, SAM 3 |
| **Geometry Engine** | Pure Rust: shoelace theorem, Green's theorem, eigenvalue decomposition, convex hull (Andrew monotone chain) |
| **FFI / IPC** | extern "C" ABI, mmap zero-copy, Mojo libmath_core.so, ctypes Python bridge |
| **API Layer** | FastAPI (Python), Axum (Rust), Streamlit dashboard |
| **Frontend** | React 19 + TypeScript + Vite, Canvas 2D rendering |
| **Deployment** | Docker, Docker Compose, Pixi (Conda-based) |
| **Integrations** | QuPath (Groovy), HALO (XML), GeoJSON standard |
| **Testing** | 108 Rust unit tests, 44 Python tests, GitHub Actions CI |

---

## Model Performance

| Model | File | AUC | F1 | Notes |
|-------|------|-----|----|-------|
| HoVerNet (fine-tuned) | `models/hovernet_finetuned.onnx` | -- | -- | 3-head (NP, HV, TP) nuclear segmentation |
| GBM Fusion Classifier | `models/nexus_fusion_classifier.joblib` | **0.98** | **0.88** | 16 geometry features + neoplastic probability |
| Topology Segmentation | `models/hovernet_topology_seg.onnx` | -- | -- | Binary material/void segmentation (Dice 0.849 on MIT TopoDiff) |

### Topology Segmentation Performance

Fine-tuned from HoVerNet (ResNet50 backbone, 37.6M params) on **MIT TopoDiff**
(30K samples, physical field inputs: strain energy / volume fraction / load):

| Metric | Value |
|--------|-------|
| **Dice** | **0.849** |
| **IoU** | **0.739** |
| **Accuracy** | 0.9915 |
| **Precision** | 0.863 |
| **Recall** | 0.838 |

Trained with combined loss (CE + Dice + Focal + Boundary) + AMP on NVIDIA L40S,
15 epochs, curriculum (2 epochs frozen backbone then full unfreeze).

See `models/hovernet_topology_meta.json` for full metadata.

### Benchmark Results (Pan-Cancer, 93 images)

- **812 cells** detected (8.7 cells/image mean, max 34)
- **265 malignant** cells predicted
- **~7 seconds** latency per image on CPU (HoVerNet 6982ms + Geometry 31ms + Classifier 45ms)
- **0.14 images/second** throughput (CPU-only)
- **AUC 0.98**, **F1 0.88**, **Accuracy 94.5%**

---

## API Quick Reference

```bash
# Health check
curl http://localhost:8810/health

# Ingest image (SAM 3 nuclear segmentation)
curl -X POST http://localhost:8810/v1/ingest/image \
  -F "file=@tissue_patch.png"

# Ingest vector annotations (XML/GeoJSON/JSON)
curl -X POST http://localhost:8810/v1/ingest/vector \
  -F "file=@annotations.xml"

# Enrich gold segment (Rust geometry + kappa curvature)
curl -X POST http://localhost:8810/v1/enrich \
  -H "Content-Type: application/json" \
  -d '{"gold_segment": {...}}'

# Export (schema validation + spatial statistics)
curl -X POST http://localhost:8810/v1/export \
  -H "Content-Type: application/json" \
  -d '{"enriched": {...}}'
```

Full API documentation: see [Technical Architecture & Scope](nexus-forge-cyto/docs/TECHNICAL_ARCHITECTURE_AND_SCOPE.md)

---

## Integrations

Nexus-Forge is designed as a **complement** to existing pathology tools, not
a replacement.

### QuPath Integration

The QuPath extension provides one-click "Analyze with Nexus-Forge" access
directly from QuPath's menu bar, with three visualization modes:
- **Clinical** — Red (Malignant), Amber (Immune), Green (Normal)
- **Heatmap** — RBF risk score as a continuous color gradient
- **Distance** — Distance-to-tumor-edge visualization

**Installation (QuPath >= 0.5.0, Java 17+):**

1. **Locate your QuPath extensions folder:**
   - Windows: `C:\Users\<username>\QuPath\extensions\`
   - macOS/Linux: `~/QuPath/extensions/`

2. **Copy the extension:**
   ```bash
   # From this repo root:
   cp -r nexus-forge-cyto/qupath-extension/ ~/QuPath/extensions/nexus-forge/
   ```

3. **Set environment variable** (or edit `getNexusHome()` in the extension):
   ```bash
   export NEXUS_FORGE_HOME=/path/to/this/repo
   # Windows PowerShell:
   $env:NEXUS_FORGE_HOME = "C:\Users\badri\cancer_project"
   ```

4. **Restart QuPath** — a new menu item appears:
   `Extensions → Nexus-Forge → Analyze with Nexus-Forge`

**Alternative: Script-based visualization** (no extension restart needed):
```groovy
# Drag & drop into QuPath, or run headless:
QuPath script -i slide.svs -s --args enriched.json \
  qupath_enriched_visualizer.groovy
```
Customize mode via system property:
```bash
# Clinical (default), heatmap, or distance
-Dnexus.vis.mode=heatmap
```

| Platform | Integration Method | Status |
|----------|-------------------|--------|
| **QuPath** | Groovy Extension (1-click analyze + visualization) | **Ready** |
| **HALO (Indica Labs)** | XML round-trip + Java SDK plugin | Planned |
| **PathAI AISight** | Marketplace partner algorithm | Planned |
| **Owkin** | Federated feature extraction engine | Planned |
| **Aiforia** | Pre-computed feature layer | Planned |

See [Integration Strategy](docs/INTEGRATION_STRATEGY.md) for the complete
complementary product roadmap.

---

## Configuration

All settings via `.env` file in the project root:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXUS_API_KEY` | (empty = open access) | FastAPI authentication key |
| `NEXUS_RUST_BRIDGE` | `auto` | Rust invocation mode: auto / shared_library / subprocess |
| `NEXUS_SAM3_MODEL_PATH` | `~/.cache/nexus-forge/sam3.pt` | SAM 3 model weights path |
| `NEXUS_GNN_WEIGHTS` | (empty) | GNN weights JSON path |
| `NEXUS_ENABLE_LLM_REPORT` | `0` | Enable LLM-based report generation |
| `QUPATH_EXE` | (empty) | QuPath executable path for integration |
| `NEXUS_DOCKER_MEM` | `8g` | Docker container memory limit |

---

## Testing

```bash
# All Rust tests (108 tests across all crates)
cargo test --all-targets

# Python tests (44 tests)
cd nexus-forge-cyto && python -m pytest tests/ -q

# Run with linting
cargo clippy -- -D warnings
cd nexus-forge-cyto && ruff check .
```

---

## Documentation

- [Technical Architecture & Scope](nexus-forge-cyto/docs/TECHNICAL_ARCHITECTURE_AND_SCOPE.md)
- [Discovered Mathematical Mappings](nexus-forge-cyto/docs/discovered_mathematical_mappings.md)
- [Integration Strategy](docs/INTEGRATION_STRATEGY.md)
- [Getting Started](docs/getting-started.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api/openapi.yaml)

---

## License

MIT OR Apache-2.0 (matches workspace `Cargo.toml`).
See [LICENSE](LICENSE) for details.

---

## Citation

If you use Nexus-Forge in your research, please cite:

```bibtex
@software{nexus_forge_cyto_2026,
  author = {ClinicalGuard Developer},
  title = {Nexus-Forge Cyto: Explainable Geometry for Digital Pathology},
  year = {2026},
  url = {https://github.com/clinicalguard/nexus-forge-cyto}
}
```
