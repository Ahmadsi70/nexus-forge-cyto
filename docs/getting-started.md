# Getting Started with Nexus-Forge Cyto

This guide gets you from zero to analyzing your first histopathology image
using Nexus-Forge's explainable geometry pipeline.

## Prerequisites

- **Rust** >= 1.75 ([rustup.rs](https://rustup.rs))
- **Python** >= 3.10
- Optional: **Docker** >= 24.0 (for containerized deployment)

## 5-Minute Quickstart

### Option A: Docker (recommended for evaluation)

```bash
# 1. Clone the repository
git clone https://github.com/clinicalguard/nexus-forge-cyto.git
cd cancer_project/nexus-forge-cyto

# 2. Start the services
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
git clone https://github.com/clinicalguard/nexus-forge-cyto.git
cd cancer_project

# 2. Build
cargo build --release

# 3. Run the pure-geometry pipeline
echo '{"cell_count":2,"cells":[[[0,0],[10,0],[10,10],[0,10]],[[20,20],[35,20],[35,35],[20,35]]]}' > /tmp/input.json

cargo run --bin nexus-core-cli -- Single --input /tmp/input.json --output /tmp/output.json

# 4. See the results
cat /tmp/output.json | python -m json.tool | head -20
```

### Option C: Python API

```bash
# 1. Clone and install
git clone https://github.com/clinicalguard/nexus-forge-cyto.git
cd cancer_project/nexus-forge-cyto
pip install -r requirements-api.txt

# 2. Start the API server
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
| SAM 3 segmentation fails | Download model: `wget https://.../sam3.pt -O ~/.cache/nexus-forge/sam3.pt` |
| "No nuclei detected" | Image may have low contrast. Try adjusting SAM 3 confidence: `NEXUS_SAM3_CONF=0.15` |
| Docker out of memory | Increase limit: `NEXUS_DOCKER_MEM=16g docker-compose up` |

## Support

- GitHub Issues: [https://github.com/clinicalguard/nexus-forge-cyto/issues](https://github.com/clinicalguard/nexus-forge-cyto/issues)
- Documentation: `docs/` directory in the repository