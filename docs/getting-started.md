# Getting Started with Nexus-Forge Cyto

This guide gets you from zero to analyzing your first histopathology image
using Nexus-Forge's explainable geometry pipeline.

## Prerequisites

- **Git LFS** ([install guide](https://git-lfs.com/)) — needed for ONNX model (145 MB)
- **Rust** >= 1.75 ([rustup.rs](https://rustup.rs))
- **Python** >= 3.10
- Optional: **Docker** >= 24.0 (for containerized deployment)
- Optional: **SAM 3** model for image → nuclei pipeline (see [Image Analysis](#image-analysis--sam-3) below)

## 5-Minute Quickstart

### Option A: Docker (recommended for evaluation)

```bash
# 1. Clone the repository (requires Git LFS)
git clone https://github.com/Ahmadsi70/nexus-forge-cyto.git
cd nexus-forge-cyto
git lfs pull

# 2. Start the services (from the nexus-forge-cyto subdirectory)
cd nexus-forge-cyto
docker-compose up -d

# 3. Open the dashboard
#    Streamlit UI → http://localhost:8501
#    FastAPI docs → http://localhost:8810/docs

# 4. Stop when done
docker-compose down
```

### Option B: Rust CLI (for developers)

```bash
# 1. Clone
git clone https://github.com/Ahmadsi70/nexus-forge-cyto.git
cd nexus-forge-cyto
git lfs pull

# 2. Build
cargo build --release

# 3. Run the pure-geometry pipeline
cargo run --release --bin nexus-core-cli -- \
  Single --input ./data/sample.json --output /tmp/output.json

# 4. See the results
cat /tmp/output.json | python -m json.tool | head -20
```

### Option C: Python API (with SAM 3)

**Requires SAM 3 model** — see [Image Analysis with SAM 3](#image-analysis-with-sam-3) below.  
The pure geometry pipeline (Option B) does not need SAM 3.

```bash
# 1. Clone and install
git clone https://github.com/Ahmadsi70/nexus-forge-cyto.git
cd nexus-forge-cyto
git lfs pull
pip install -r nexus-forge-cyto/requirements-api.txt

# 2. Start the API server
cd nexus-forge-cyto
NEXUS_RUST_BRIDGE=subprocess PYTHONPATH=services python services/api_service/main.py

# 3. In another terminal, upload an image
curl -X POST http://localhost:8810/v1/ingest/image \
  -F "file=@your_hne_patch.png" \
  | python -m json.tool
```

## Your First Analysis

### Step 1: Prepare Input

Nexus-Forge accepts cell polygon annotations in several formats:

| Format | Extension | Tool |
|--------|-----------|------|
| MoNuSeg / ImageScope XML | `.xml` | HALO, QuPath |
| GeoJSON FeatureCollection | `.geojson` | QuPath, most GIS tools |
| JSON polygons | `.json` | Custom, PanNuke-format |
| CSV vertex tables | `.csv` | Microsoft Excel, manual |

### Step 2: Run the Pipeline

**Via HTTP API:**
```bash
# Upload XML/GeoJSON annotations
curl -X POST http://localhost:8810/v1/ingest/vector \
  -F "file=@annotations.xml" \
  -o gold.json

# Enrich with geometry + spatial topology
curl -X POST http://localhost:8810/v1/enrich \
  -H "Content-Type: application/json" \
  -d @gold.json \
  -o enriched.json

# Export final results
curl -X POST http://localhost:8810/v1/export \
  -H "Content-Type: application/json" \
  -d @enriched.json \
  -o final.json
```

**Via Rust CLI:**
```bash
cargo run --bin nexus-forge-cyto-batch -- \
  --input-dir ./my_annotations/ \
  --output-dir ./enriched_output/
```

### Step 3: View Results

The enriched JSON contains:

- `cells` — 128-point polygon rings per cell
- `clinical` — "Malignant" or "Normal" per cell
- `spatial_features` — per-cell metrics (area, circularity, eccentricity, kNN density, RBF risk, distance to tumor edge)
- `spatial_analysis` — global metrics (kNN k, RBF sigma)
- `innovations` — 14 mathematical features (spectral embedding, density ratios, tissue IQ, etc.)

### Step 4: Visualize in QuPath

Drag the enriched JSON into QuPath along with your slide image, then run:

```
Extensions → Nexus-Forge → Import Enriched Results
```

Red = Malignant, Blue = Normal. Measurements appear in the Measurements tab.

## Next Steps

- Read the [Technical Architecture](nexus-forge-cyto/docs/TECHNICAL_ARCHITECTURE_AND_SCOPE.md) for pipeline details
- See [Integration Strategy](docs/INTEGRATION_STRATEGY.md) for QuPath, HALO, PathAI, Owkin, and Aiforia
- Check [Deployment Guide](docs/deployment.md) for production setup
- Explore the [API Reference](docs/api/openapi.yaml)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `cargo build` fails | Ensure Rust >= 1.75: `rustup update stable` |
| Python imports fail | Run from `nexus-forge-cyto/` with `PYTHONPATH=services` |
| SAM 3 segmentation fails | See [Image Analysis with SAM 3](#image-analysis--sam-3) below |
| "No nuclei detected" | Image may have low contrast. Try adjusting SAM 3 confidence: `NEXUS_SAM3_CONF=0.15` |
| Docker out of memory | Increase limit: `NEXUS_DOCKER_MEM=16g docker-compose up` |

---

## Image Analysis with SAM 3

For full image → nuclei → geometry pipeline, you need the **SAM 3** model
(~2.4 GB) from Facebook Research / Ultralytics:

```bash
# 1. Get a HuggingFace token
#    https://huggingface.co/settings/tokens  →  Create "read" token

# 2. Export your token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 3. Run the setup script
python scripts/setup_sam3.py

# 4. Verify it works
python -c "from api_service.sam3_segment import _sam3_model_path; print('SAM3 path:', _sam3_model_path())"
PYTHONPATH=services
```

The model is downloaded to `~/.cache/nexus-forge/sam3.pt` by default.
Override with `NEXUS_SAM3_MODEL_PATH` environment variable.

### SAM 3 Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXUS_SAM3_MODEL_PATH` | `~/.cache/nexus-forge/sam3.pt` | Path to sam3.pt weights |
| `NEXUS_SAM3_CONF` | `0.15` | Detection confidence threshold |
| `NEXUS_SAM3_NMS_IOU` | `0.30` | Non-maximum suppression IoU threshold |
| `NEXUS_SAM3_BOX_CHUNK` | `48` | Max boxes per SAM3 inference call |

---

## HoVerNet Nuclear Segmentation (Pre-Trained Weights)

The [original HoVerNet](https://github.com/vqdang/hover_net) fold-12
nuclear segmentation weights (`hovernet_finetuned.onnx`, ~151 MB) are
**not included** in this repository due to licensing restrictions.

To use the full AI + Geometry pipeline, download the weights separately:

```bash
# Option A: Download from the original HoVerNet repository
# See: https://github.com/vqdang/hover_net/releases
# Place at: models/hovernet_finetuned.onnx

# Option B: Use the topology model (included via Git LFS)
# python run_topology_inference.py --info
```

The **topology segmentation model** (`hovernet_topology_seg.onnx`) IS
included and works on CPU — no GPU or external download needed.

---

## Mojo κ (Curvature Acceleration)

The **Mojo κ engine** provides SIMD-accelerated curvature computation
for large batches. It is optional — the Rust fallback (`κ=0` / normal)
is always available in the binary distribution.

```bash
# Mojo is NOT required for any pipeline feature.
# It only improves throughput on very large batches (>10K cells).

# If you have Mojo 0.26.1.0 installed:
cd mojo_core
mojo build kappa_engine.mojo
# The resulting .so is auto-detected by the Rust FFI bridge.
```

### Mojo Status

| Feature | With Mojo | Without Mojo (Rust fallback) |
|---------|-----------|------------------------------|
| Curvature (κ) | Full computation | `κ=0`, flagged as "normal" |
| Throughput | ~3× faster for large batches | Same for <1K cells |
| Installation | Mojo SDK (free tier available) | None needed |

## Support

- GitHub Issues: [https://github.com/Ahmadsi70/nexus-forge-cyto/issues](https://github.com/Ahmadsi70/nexus-forge-cyto/issues)
- Documentation: `docs/` directory in the repository