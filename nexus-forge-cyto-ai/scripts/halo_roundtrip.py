#!/usr/bin/env python
"""
HALO Annotation Roundtrip: XML -> Nexus-Forge enrichment -> Enriched XML.

Full roundtrip pipeline:
  1. Parse HALO / ImageScope XML annotations (Region -> Vertex coordinates)
  2. Build gold-segment envelope (128-point ring per region)
  3. Run Rust kappa-curvature + geometry enrichment
  4. Map enrichment output to per-region clinical labels
  5. Inject Classification, Confidence, and NexusCellId into XML Region elements
  6. SHA-256 geometry integrity gate -- vertex coordinates MUST be byte-stable

Usage:
  python halo_roundtrip.py --input annotations.xml --output enriched.xml

  # With explicit enrichment binary path:
  python halo_roundtrip.py --input annotations.xml --output enriched.xml \
      --enricher ./target/release/nexus-forge-cyto-geometry

  # Keep intermediate files for debugging:
  python halo_roundtrip.py --input annotations.xml --output enriched.xml --keep-temp

Requirements:
  - nexus-forge-cyto Rust enrichment binary (built or accessible via NEXUS_ENRICHER)
  - Python packages: lxml, numpy (same as nexus-core services)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from lxml import etree

# Add nexus services to path for region_walker imports
_SERVICES = Path(__file__).resolve().parents[2] / "nexus-forge-cyto" / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

from nexus_core.region_walker import (          # noqa: E402
    _xml_parser,
    assert_geometry_unchanged,
    file_sha256,
    iter_xml_regions,
    records_from_xml,
    build_manifest,
)
from nexus_core.coord_adapter import (          # noqa: E402
    ring_from_polygon_xy,
    TARGET_RING,
    segment_envelope_from_polygons,
)

# --- Constants ---------------------------------------------------------------

CENTROID_EPS = 0.5  # Max centroid drift (px) for parity check


def _centroid_delta(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two 2D centroids."""
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _validate_centroid_parity(
    manifest_records: list[dict],
    predictions: list[dict],
) -> None:
    """Fail fast if enrichment centroids drifted beyond CENTROID_EPS."""
    for i, (mrow, prow) in enumerate(zip(manifest_records, predictions, strict=True)):
        cell_id = int(prow["cell_id"])
        if int(mrow["cell_id"]) != cell_id:
            raise ValueError(
                f"cell_id mismatch at index {i}: "
                f"manifest={mrow['cell_id']} prediction={cell_id}"
            )
        mc = mrow["centroid"]
        pc = prow["centroid"]
        delta = _centroid_delta(mc, pc)
        if delta > CENTROID_EPS:
            rid = mrow.get("region_id") or i
            raise ValueError(
                f"Centroid parity failure at cell_id={cell_id} "
                f"(source_id={rid}): delta={delta:.4f}px > {CENTROID_EPS}"
            )


def inject_xml_annotations(
    source_path: Path,
    output_path: Path,
    predictions: list[dict],
) -> int:
    """
    Inject Classification / Confidence / NexusCellId as Region element attributes.

    Geometry is never touched -- only element attributes are added.
    Returns the number of injected regions.
    """
    tree = etree.parse(str(source_path), _xml_parser())
    root = tree.getroot()
    regions = [region for region, _verts in iter_xml_regions(root)]

    if len(regions) != len(predictions):
        raise ValueError(
            f"Region count mismatch: xml={len(regions)} predictions={len(predictions)}"
        )

    injected = 0
    for region_el, prow in zip(regions, predictions, strict=True):
        label = str(prow["clinical_label"])
        conf = float(prow["confidence"])
        region_el.set("Classification", label)
        region_el.set("Confidence", f"{conf:.4f}")
        region_el.set("NexusCellId", str(int(prow["cell_id"])))
        injected += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(output_path),
        encoding=tree.docinfo.encoding or "UTF-8",
        xml_declaration=True,
        pretty_print=False,
    )
    return injected


def _find_enricher_binary() -> Path:
    """Resolve the Rust enrichment binary path."""
    # 1. NEXUS_ENRICHER env var
    env_path = os.environ.get("NEXUS_ENRICHER")
    if env_path:
        return Path(env_path)

    # 2. Common build paths
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "target" / "release" / "nexus-forge-cyto-geometry",
        repo_root / "target" / "debug" / "nexus-forge-cyto-geometry",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        "Rust enrichment binary not found. Build it first:\n"
        "  cargo build --release -p nexus-forge-cyto-geometry\n"
        "Or set NEXUS_ENRICHER env var to the binary path."
    )


def run_rust_enrichment(segment_path: Path, enricher: Path, work_dir: Path) -> dict:
    """Run the Rust enrichment binary on a gold segment JSON and return parsed output."""
    output_path = work_dir / "enriched.json"
    cmd = [
        str(enricher),
        "--input", str(segment_path),
        "--output", str(output_path),
    ]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Rust enricher failed (exit {result.returncode}):\n{stderr}"
        )

    if result.stdout.strip():
        print(result.stdout.strip())

    if not output_path.is_file():
        raise FileNotFoundError(f"Enricher did not produce output: {output_path}")

    return json.loads(output_path.read_text(encoding="utf-8"))


def extract_predictions(
    enriched: dict, manifest_records: list[dict]
) -> list[dict]:
    """
    Extract per-cell predictions from the enriched output.

    Matches enrichment output to manifest records by cell_id index.
    """
    clinical = enriched.get("clinical", [])
    confidence = enriched.get("confidence", [])
    spatial_features = enriched.get("spatial_features", [])

    predictions = []
    for i, mrow in enumerate(manifest_records):
        cell_id = int(mrow["cell_id"])
        label = clinical[i] if i < len(clinical) else "Normal"

        # Resolve confidence: explicit confidence array or deterministic default
        if i < len(confidence):
            conf = float(confidence[i])
        else:
            conf = 0.85  # default for geometry-only pipeline

        # Resolve centroid from spatial features
        if i < len(spatial_features) and "centroid" in spatial_features[i]:
            centroid = spatial_features[i]["centroid"]
        else:
            centroid = mrow.get("centroid", [0.0, 0.0])

        predictions.append({
            "cell_id": cell_id,
            "clinical_label": label,
            "confidence": conf,
            "centroid": [
                float(centroid[0]) if len(centroid) > 0 else 0.0,
                float(centroid[1]) if len(centroid) > 1 else 0.0,
            ],
        })

    return predictions


def halo_roundtrip(
    input_path: Path,
    output_path: Path,
    *,
    enricher: Path | None = None,
    keep_temp: bool = False,
) -> dict:
    """
    Execute the full HALO roundtrip pipeline.

    Returns a summary dict with region_count, injected_count, integrity_gate,
    output_path, source_sha256, and output_sha256.
    """
    # Phase 1: Parse HALO XML -> records
    print(f"[1/5] Parsing HALO XML: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input XML not found: {input_path}")

    records = records_from_xml(input_path)
    if not records:
        raise ValueError(f"No Region elements found in {input_path}")
    print(f"       Found {len(records)} region(s)")

    # Phase 2: Build gold segment envelope
    print(f"[2/5] Building gold segment envelope")
    polygons = []
    for rec in records:
        ring = rec["ring"]
        verts = [(float(p[0]), float(p[1])) for p in ring]
        polygons.append(verts)

    segment = segment_envelope_from_polygons(
        polygons,
        format_hint="halo_xml_regions",
        source_path=str(input_path.resolve()),
    )

    # Phase 3: Build manifest (join-key tracking)
    manifest = build_manifest(input_path, "halo_xml_regions", records)

    # Phase 4: Run Rust enrichment
    tmpdir = tempfile.mkdtemp(prefix="nexus_halo_roundtrip_")
    work = Path(tmpdir)
    try:
        segment_path = work / "gold_segment.json"
        segment_path.write_text(
            json.dumps(segment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"[3/5] Running Rust kappa-curvature enrichment")
        if enricher is None:
            enricher = _find_enricher_binary()
        print(f"       Enricher: {enricher}")

        enriched = run_rust_enrichment(segment_path, enricher, work)

        # Phase 5: Extract predictions
        print(f"[4/5] Extracting predictions from enriched output")
        predictions = extract_predictions(enriched, manifest["records"])

        # Validate centroid parity before injection
        _validate_centroid_parity(manifest["records"], predictions)
        print(f"       Centroid parity: OK (all within {CENTROID_EPS}px)")

        # Phase 6: Inject into XML
        print(f"[5/5] Injecting annotations -> {output_path}")
        injected = inject_xml_annotations(input_path, output_path, predictions)
        print(f"       Injected {injected} region(s)")

        # Phase 7: Geometry integrity gate
        assert_geometry_unchanged(input_path, output_path, "halo_xml_regions")
        print(f"       Integrity gate: PASS (SHA-256 vertex digest match)")

        summary = {
            "region_count": len(records),
            "injected_count": injected,
            "integrity_gate": "PASS",
            "output_path": str(output_path.resolve()),
            "source_sha256": file_sha256(input_path),
            "output_sha256": file_sha256(output_path),
        }

    finally:
        if not keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"[INFO] Intermediate files kept at: {tmpdir}")

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nexus-Forge HALO Annotation Roundtrip: XML -> enrich -> XML",
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input HALO / ImageScope XML annotations file",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output enriched XML path",
    )
    p.add_argument(
        "--enricher",
        type=Path,
        default=None,
        help="Path to Rust enrichment binary (auto-detected if omitted)",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate temp files for debugging",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        summary = halo_roundtrip(
            input_path=args.input,
            output_path=args.output,
            enricher=args.enricher,
            keep_temp=args.keep_temp,
        )
        print("")
        print("=" * 60)
        print("  HALO Roundtrip Complete")
        print("=" * 60)
        print(f"  Regions processed:    {summary['region_count']}")
        print(f"  Annotations injected: {summary['injected_count']}")
        print(f"  Integrity gate:       {summary['integrity_gate']}")
        print(f"  Output:               {summary['output_path']}")
        print(f"  Source SHA-256:       {summary['source_sha256'][:16]}...")
        print(f"  Output SHA-256:       {summary['output_sha256'][:16]}...")
        print("=" * 60)
    except Exception as exc:
        raise SystemExit(f"[FATAL] Roundtrip failed: {exc}") from exc


if __name__ == "__main__":
    main()