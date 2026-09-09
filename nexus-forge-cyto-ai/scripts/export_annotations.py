"""
Template-preserving annotation exporter: inject GNN classifications into source XML/GeoJSON.

Why: clinical teams require vendor-native annotation files with AI labels added as metadata
only — geometry and hierarchy must remain byte-stable aside from new attributes/properties.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lxml import etree

_SERVICES = Path(__file__).resolve().parents[1] / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))
from nexus_core.region_walker import (  # noqa: E402
    file_sha256,
    _xml_parser,
    assert_geometry_unchanged,
    iter_xml_regions,
)
from nexus_core.runtime_paths import production_output_dir  # noqa: E402

CENTROID_EPS = 0.5


def _load_predictions(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "records" in raw:
        rows = raw["records"]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise TypeError(f"Expected list or dict with 'records' in {path}")
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    return rows


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"source_path", "source_format", "source_sha256", "record_count", "records"}
    missing = required.difference(manifest.keys())
    if missing:
        raise KeyError(f"Manifest missing keys: {sorted(missing)}")
    return manifest


def _validate_manifest_source(source_path: Path, manifest: dict) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source annotation not found: {source_path}")
    digest = file_sha256(source_path)
    if digest != manifest["source_sha256"]:
        raise ValueError(
            "Source file changed since ingest: SHA-256 mismatch "
            f"(expected={manifest['source_sha256'][:16]}… got={digest[:16]}…)"
        )


def _centroid_delta(a: list[float], b: list[float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _validate_centroid_parity(
    manifest_records: list[dict], predictions: list[dict]
) -> None:
    for i, (mrow, prow) in enumerate(zip(manifest_records, predictions, strict=True)):
        cell_id = int(prow["cell_id"])
        if int(mrow["cell_id"]) != cell_id:
            raise ValueError(
                f"cell_id mismatch at index {i}: manifest={mrow['cell_id']} prediction={cell_id}"
            )
        mc = mrow["centroid"]
        pc = prow["centroid"]
        delta = _centroid_delta(mc, pc)
        if delta > CENTROID_EPS:
            rid = mrow.get("region_id") or mrow.get("feature_id") or i
            raise ValueError(
                f"Centroid parity failure at cell_id={cell_id} (source_id={rid}): "
                f"delta={delta:.4f}px > {CENTROID_EPS}"
            )


def export_xml_annotations(
    source_path: Path,
    output_path: Path,
    predictions: list[dict],
    manifest: dict,
) -> int:
    """Inject Classification/Confidence attributes on Region elements; geometry untouched."""
    tree = etree.parse(str(source_path), _xml_parser())
    root = tree.getroot()
    regions = [region for region, _verts in iter_xml_regions(root)]
    if len(regions) != len(predictions):
        raise ValueError(
            f"Region count mismatch: xml={len(regions)} predictions={len(predictions)}"
        )
    _validate_centroid_parity(manifest["records"], predictions)

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


def export_geojson_annotations(
    source_path: Path,
    output_path: Path,
    predictions: list[dict],
    manifest: dict,
) -> int:
    """Inject Classification into Feature properties; geometry coordinates unchanged."""
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    feats = payload.get("features", [])
    eligible = [f for f in feats if isinstance(f, dict) and isinstance(f.get("geometry"), dict)]
    if len(eligible) != len(predictions):
        raise ValueError(
            f"Feature count mismatch: geojson={len(eligible)} predictions={len(predictions)}"
        )
    _validate_centroid_parity(manifest["records"], predictions)

    # Build a lookup from sequential source_index (document order) → prediction.
    # Manifest records enumerate features in document order with cell_id mapping.
    manifest_recs = manifest["records"]
    pred_by_cell_id = {int(p["cell_id"]): p for p in predictions}
    injected = 0
    pred_idx = 0
    for feat in feats:
        if not isinstance(feat, dict) or not isinstance(feat.get("geometry"), dict):
            continue
        # Use manifest's source_index to resolve the correct prediction cell_id.
        if pred_idx < len(manifest_recs):
            cell_id = int(manifest_recs[pred_idx]["cell_id"])
        else:
            cell_id = pred_idx  # fallback: assume sequential
        prow = pred_by_cell_id.get(cell_id)
        if prow is None:
            raise KeyError(
                f"Missing prediction for feature index {pred_idx} (cell_id={cell_id})"
            )
        props = feat.setdefault("properties", {})
        if not isinstance(props, dict):
            props = {}
            feat["properties"] = props
        props["Classification"] = str(prow["clinical_label"])
        props["Confidence"] = float(prow["confidence"])
        props["NexusCellId"] = int(prow["cell_id"])
        injected += 1
        pred_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return injected


def export_annotations(
    source_path: Path,
    predictions_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict:
    """Run format-specific export plus geometry integrity gate."""
    manifest = _load_manifest(manifest_path)
    predictions = _load_predictions(predictions_path)
    if len(predictions) != int(manifest["record_count"]):
        raise ValueError(
            f"Prediction/manifest count mismatch: predictions={len(predictions)} "
            f"manifest={manifest['record_count']}"
        )

    resolved_source = source_path.resolve()
    manifest_source = Path(manifest["source_path"]).resolve()
    if resolved_source != manifest_source:
        print(
            f"[WARN] source_path differs from manifest "
            f"(cli={resolved_source} manifest={manifest_source}); validating SHA-256 only"
        )
    _validate_manifest_source(resolved_source, manifest)

    fmt = str(manifest["source_format"])
    if fmt in {"halo_xml_regions", "monuseg_xml"}:
        count = export_xml_annotations(resolved_source, output_path, predictions, manifest)
    elif fmt == "geojson_features":
        count = export_geojson_annotations(resolved_source, output_path, predictions, manifest)
    else:
        raise ValueError(f"Unsupported manifest source_format: {fmt!r}")

    assert_geometry_unchanged(resolved_source, output_path, fmt)
    return {
        "source_format": fmt,
        "injected_count": count,
        "output_path": str(output_path.resolve()),
        "integrity_gate": "PASS",
    }


def _parse_args() -> argparse.Namespace:
    default_out = production_output_dir()
    p = argparse.ArgumentParser(description="Inject GNN classifications into source annotations")
    p.add_argument("--source", type=Path, required=True, help="Original XML or GeoJSON file")
    p.add_argument(
        "--predictions",
        type=Path,
        default=default_out / "node_predictions.json",
        help="Per-node predictions JSON",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=default_out / "annotation_index_manifest.json",
        help="Annotation index manifest from ingest",
    )
    p.add_argument("--output", type=Path, required=True, help="Annotated output path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        summary = export_annotations(
            source_path=args.source,
            predictions_path=args.predictions,
            manifest_path=args.manifest,
            output_path=args.output,
        )
        print(f"[PASS] integrity_gate={summary['integrity_gate']}")
        print(f"[DONE] injected={summary['injected_count']} format={summary['source_format']}")
        print(f"[DONE] output={summary['output_path']}")
    except Exception as exc:
        raise SystemExit(f"[FATAL] export_annotations failed: {exc}") from exc


if __name__ == "__main__":
    main()
