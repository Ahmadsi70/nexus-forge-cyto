# Nexus-Forge Cyto — QuPath Extension

One-click explainable geometry analysis for digital pathology annotations.

## What It Does

Adds transparent morphometrics, spatial topology, and auditable malignancy
scoring to QuPath cell annotations, without requiring GPU, cloud, or black-box
AI models.

## Features

- **One-click analysis** — Select annotations → Extensions → Nexus-Forge → Analyze
- **19 numeric features** per cell injected into QuPath Measurements table:
  Area, Perimeter, Circularity, Eccentricity, Solidity, Convexity, Hu Moments (7),
  kNN Density, RBF Risk Score, Distance to Tumor Edge
- **Color-coded overlay** — Red = Malignant, Blue = Normal
- **Deterministic & Explainable** — Published weights, bit-for-bit identical
  outputs, fully auditable

## Quick Install

1. Copy the `qupath-extension/` directory into your QuPath extensions folder:
   - **Linux/macOS:** `~/QuPath/extensions/nexus-forge/`
   - **Windows:** `C:\Users\<YourName>\QuPath\extensions\nexus-forge\`

2. Set the `NEXUS_FORGE_HOME` environment variable to your nexus-forge-cyto
   checkout directory.

3. Restart QuPath. A new menu item appears under **Extensions → Nexus-Forge**.

## Requirements

- QuPath >= 0.5.0
- Java 17+
- Python >= 3.10 with the `nexus-forge-cyto` services installed

## Usage

1. Open a slide image in QuPath
2. Detect cells using QuPath's built-in cell detection or the StarDist extension
3. Select the annotations you want to analyze
4. Click **Extensions → Nexus-Forge → Analyze with Nexus-Forge**
5. Results appear as color-coded annotations with per-cell metrics in the
   Measurements tab

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Nexus-Forge not found" | Set `NEXUS_FORGE_HOME` env var to your checkout path |
| No annotations visible | Run cell detection first (Analyze → Cell detection → Cell detection) |
| Analysis hangs | Check that Python >= 3.10 is installed and on PATH |
| Error in measurement values | Ensure annotations are closed polygons with >= 3 points |

## License

MIT OR Apache-2.0 — matches the parent project.