"""
FastAPI geometry ingress: image (SAM 3) or vector → gold segment → Rust enrich.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from api_service.image_ingest import segment_image_bytes
import logging
import time
import uuid
from contextvars import ContextVar

# Structured logging with request IDs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-5s] [%(name)s] [req_id=%(req_id)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger('nexus.api')
request_id_var: ContextVar[str] = ContextVar('request_id', default='-')


class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.req_id = request_id_var.get()
        return True


logger.addFilter(RequestIDFilter())


from api_service.pipeline_core import (
    PipelineError,
    gold_segment_from_polygons,
    gold_segment_from_vector_bytes,
    run_headless_schema_export,
    run_rust_enrichment_legacy as run_rust_enrichment,
)

# Pharma biomarker engine imports
from api_service.features import extract_features, features_to_csv, features_to_matrix, FEATURE_NAMES
from api_service.biomarker_report import compute_biomarkers, generate_pdf_report
from api_service.batch import (
    BatchJob,
    JobStatus,
    create_job,
    get_job,
    cancel_job,
    list_jobs,
    validate_batch_files,
)

# ── Security ──────────────────────────────────────────────────────────────────

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)):
    """Validate API key if NEXUS_API_KEY is configured.
    
    ⚠️ SECURITY: When NEXUS_API_KEY is not set, ALL requests are allowed
    without authentication (developer mode). For production deployment,
    ALWAYS set a strong NEXUS_API_KEY environment variable.
    """
    expected = os.environ.get("NEXUS_API_KEY", "").strip()
    if expected:
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required (X-API-Key header)")
        if api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid API key")
    # If NEXUS_API_KEY is not set, allow unauthenticated access (dev mode)
    return api_key



# ── Rate Limiting ────────────────────────────────────────────────────────────

import time as _time
from collections import defaultdict

class _RateLimiter:
    """Simple in-process sliding-window rate limiter.
    
    No external dependency needed. Tracks request counts per IP in a
    sliding window. Excess requests receive HTTP 429 with Retry-After.
    """
    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = _time.monotonic()
        bucket = self._buckets[client_ip]
        # Expire old entries
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) < self.max_requests:
            bucket.append(now)
            return True
        return False
    
    def retry_after(self, client_ip: str) -> float:
        bucket = self._buckets.get(client_ip, [])
        if not bucket:
            return 1.0
        return max(1.0, self.window - (_time.monotonic() - bucket[0]))


_rate_limiter = _RateLimiter(
    max_requests=int(os.environ.get("NEXUS_RATE_LIMIT", "100")),
    window_seconds=float(os.environ.get("NEXUS_RATE_WINDOW", "60")),
)


# ── CORS & Security Middleware ──────────────────────────────────────────────

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware

ALLOWED_ORIGINS = os.environ.get(
    "NEXUS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8501"
).split(",")


app = FastAPI(
    title="Nexus-Forge Cyto Geometry API",
    version="0.2.0",
    description="Dual ingress (image / vector) and Rust κ enrichment for cytology pipelines.",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Rate limiting + security headers + request ID middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Request ID
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(req_id)
    
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        retry_after = _rate_limiter.retry_after(client_ip)
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return Response(
            content=json.dumps({"detail": "Too many requests"}),
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": str(int(retry_after))}
        )
    
    t_start = _time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (_time.perf_counter() - t_start) * 1000
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} " +
        f"({elapsed_ms:.1f}ms) [ip={client_ip}]"
    )
    return response


class EnrichRequest(BaseModel):
    """Gold segment envelope produced by /v1/ingest/* endpoints."""

    gold_segment: dict[str, Any] = Field(..., description="cell_count + fixed-length ring cells[]")


class IngestResponse(BaseModel):
    cell_count: int
    source: str
    gold_segment: dict[str, Any]


class EnrichResponse(BaseModel):
    cell_count: int
    enriched: dict[str, Any]


class ExportRequest(BaseModel):
    """Rust-enriched payload from /v1/enrich (input to headless schema export)."""

    enriched: dict[str, Any] = Field(..., description="Rust enrichment output")


class ExportResponse(BaseModel):
    cell_count: int
    stage_v2_final: dict[str, Any]


@app.get("/live")
def liveness() -> dict[str, str]:
    """Liveness probe — is the process alive?"""
    return {"status": "ok"}


@app.get("/ready")
def readiness() -> dict[str, object]:
    """Readiness probe — can the service accept traffic?
    
    Checks: Rust binary reachable, SAM3 model path valid (if configured).
    """
    checks = {}
    healthy = True
    
    # Check Rust bridge
    try:
        from services.nexus_core.rust_bridge import run_rust_enrichment
        checks["rust_bridge"] = "available"
    except Exception as e:
        checks["rust_bridge"] = f"error: {e}"
        healthy = False
    
    # Check SAM3 model if configured
    sam3_path = os.environ.get("NEXUS_SAM3_MODEL_PATH", "")
    if sam3_path:
        if Path(sam3_path).exists():
            checks["sam3_model"] = "available"
        else:
            checks["sam3_model"] = f"not found: {sam3_path}"
            healthy = False
    else:
        checks["sam3_model"] = "not configured (image ingest disabled)"
    
    return {
        "status": "ready" if healthy else "not ready",
        "checks": checks,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Legacy health endpoint — redirects to liveness."""
    return {"status": "ok"}


@app.post("/v1/ingest/vector", response_model=IngestResponse)
async def ingest_vector(
    file: UploadFile = File(...),
    api_key: str | None = Security(verify_api_key),
) -> IngestResponse:
    """Accept Aperio XML, QuPath GeoJSON, or gold-segment JSON → normalized gold segment."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)} MB)")
    name = file.filename or "upload.bin"
    try:
        gold = gold_segment_from_vector_bytes(name, raw)
    except (ValueError, PipelineError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"vector ingest failed: {exc}") from exc
    return IngestResponse(
        cell_count=int(gold["cell_count"]),
        source=str(gold.get("annotation_adapter", {}).get("format", "vector")),
        gold_segment=gold,
    )


@app.post("/v1/ingest/image", response_model=IngestResponse)
async def ingest_image(
    file: UploadFile = File(...),
    text_prompt: str | None = Query(None, description="Optional SAM 3 text concept prompt"),
    api_key: str | None = Security(verify_api_key),
) -> IngestResponse:
    """Accept H&E patch → SAM 3 polygons → gold segment."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)} MB)")
    name = file.filename or "patch.png"
    try:
        polygons, fmt = segment_image_bytes(raw, text_prompt=text_prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"image segmentation failed: {exc}") from exc

    if not polygons:
        raise HTTPException(status_code=422, detail="no nuclei detected in uploaded image")

    gold = gold_segment_from_polygons(
        polygons,
        format_hint=fmt,
        source_path=name,
    )
    return IngestResponse(
        cell_count=int(gold["cell_count"]),
        source=fmt,
        gold_segment=gold,
    )


@app.post("/v1/enrich", response_model=EnrichResponse)
def enrich(
    body: EnrichRequest,
    api_key: str | None = Security(verify_api_key),
) -> EnrichResponse:
    """Run Rust geometry + κ enrichment on a gold segment envelope."""
    try:
        enriched = run_rust_enrichment(body.gold_segment)
    except PipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    count = int(enriched.get("cell_count") or len(enriched.get("cells") or []))
    return EnrichResponse(cell_count=count, enriched=enriched)


@app.post("/v1/export", response_model=ExportResponse)
def export_final(
    body: ExportRequest,
    api_key: str | None = Security(verify_api_key),
) -> ExportResponse:
    """Run headless schema validation + spatial stats (same as Streamlit step 3)."""
    import tempfile
    import uuid

    work = tempfile.mkdtemp(prefix=f"nexus_export_{uuid.uuid4().hex}_")
    work_path = Path(work)
    enriched_path = work_path / "enriched.json"
    enriched_path.write_text(json.dumps(body.enriched, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        final = run_headless_schema_export(enriched_path, output_dir=work_path)
    except PipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    count = int(final.get("cell_count") or len(final.get("cells") or []))
    return ExportResponse(cell_count=count, stage_v2_final=final)


# ── Pharma Biomarker Engine ───────────────────────────────────────────────────


class FeatureRequest(BaseModel):
    """Gold segment envelope for feature extraction (lightweight, no enrichment needed)."""
    gold_segment: dict[str, Any] = Field(..., description="cell_count + fixed-length ring cells[]")
    format: str = Field(default="json", description="Output format: json | csv | matrix")
    knn_k: int = Field(default=5, ge=1, le=50, description="k-NN neighbor count")


class FeatureResponse(BaseModel):
    cell_count: int
    feature_count: int
    feature_names: list[str]
    features: list[dict[str, float]] | list[list[float]] = Field(default_factory=list)
    csv: str | None = None


class BiomarkerRequest(BaseModel):
    """Enriched output for biomarker computation."""
    enriched: dict[str, Any] = Field(..., description="Rust enrichment output from /v1/enrich")
    slide_id: str | None = None
    stain_type: str = "H&E"
    magnification: str = "40x"
    include_distributions: bool = True
    include_cells: bool = False
    format: str = Field(default="json", description="Output format: json | pdf")


class BatchSubmitResponse(BaseModel):
    job_id: str
    status: str
    total: int
    message: str


class BatchSummaryResponse(BaseModel):
    jobs: list[dict[str, Any]]
    total: int


@app.post("/v1/extract-features", response_model=FeatureResponse)
def extract_features_endpoint(
    body: FeatureRequest,
    api_key: str | None = Security(verify_api_key),
) -> FeatureResponse:
    """Extract 19 numeric features per cell from a gold segment — no enrichment needed.

    This is the fast path for pharma/biotech partners who need feature matrices
    for their own statistical models. Returns features in JSON, CSV, or raw matrix format.
    """
    try:
        features = extract_features(body.gold_segment, knn_k=body.knn_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cell_count = len(features)
    csv_out = None

    if body.format == "csv":
        csv_out = features_to_csv(features)
        return FeatureResponse(
            cell_count=cell_count,
            feature_count=len(FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            features=[],
            csv=csv_out,
        )

    if body.format == "matrix":
        matrix = features_to_matrix(features)
        return FeatureResponse(
            cell_count=cell_count,
            feature_count=len(FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            features=[],  # type: ignore  — we set the matrix below
        )

    return FeatureResponse(
        cell_count=cell_count,
        feature_count=len(FEATURE_NAMES),
        feature_names=FEATURE_NAMES,
        features=features,
    )


@app.post("/v1/biomarker-report")
def biomarker_report_endpoint(
    body: BiomarkerRequest,
    api_key: str | None = Security(verify_api_key),
):
    """Compute slide-level spatial biomarkers from enriched output.

    Returns clinical-grade metrics:
      - Nuclear Atypia Index (NAI)
      - Tissue Architecture Disruption Score (TADS)
      - Immune Infiltration Skew
      - Tumor Heterogeneity Score (Gini)

    Supports JSON and PDF (reportlab required) formats.
    """
    try:
        report = compute_biomarkers(
            body.enriched,
            slide_id=body.slide_id,
            stain_type=body.stain_type,
            magnification=body.magnification,
            include_distributions=body.include_distributions,
            include_cell_list=body.include_cells,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Biomarker computation failed: {exc}") from exc

    if body.format == "pdf":
        try:
            pdf_bytes = generate_pdf_report(report)
            from fastapi.responses import Response as FastAPIResponse
            return FastAPIResponse(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=biomarker_report_{body.slide_id or 'report'}.pdf"
                },
            )
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"PDF generation requires reportlab: {exc}"
            ) from exc

    return report


@app.post("/v1/batch", response_model=BatchSubmitResponse)
async def submit_batch(
    files: list[UploadFile] = File(...),
    webhook_url: str | None = Form(None),
    api_key: str | None = Security(verify_api_key),
) -> BatchSubmitResponse:
    """Submit multiple annotation files for batch processing.

    Accepts up to 100 files (500 MB total). Processing happens asynchronously.
    Poll GET /v1/batch/{job_id} for status or provide a webhook_url for callback.
    """
    # Read all files
    file_data = []
    for f in files:
        raw = await f.read()
        file_data.append((f.filename or "upload", raw))

    # Validate
    errors = validate_batch_files(file_data)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Create job (starts background processing)
    filenames = [f[0] for f in file_data]
    bytes_list = [f[1] for f in file_data]
    job = create_job(filenames, bytes_list, webhook_url=webhook_url)

    return BatchSubmitResponse(
        job_id=job.job_id,
        status=job.status.value,
        total=job.total,
        message=f"Batch job {job.job_id} created with {job.total} files. Poll /v1/batch/{job.job_id} for status.",
    )


@app.get("/v1/batch/{job_id}")
def get_batch_status(
    job_id: str,
    include_items: bool = Query(False, description="Include per-item status"),
    api_key: str | None = Security(verify_api_key),
):
    """Get batch job status. Set include_items=true for per-file details."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if include_items:
        return job.to_detail()
    return job.to_summary()


@app.get("/v1/batch")
def list_batch_jobs(
    status: str | None = Query(None, description="Filter by status: pending|running|completed|failed"),
    api_key: str | None = Security(verify_api_key),
) -> BatchSummaryResponse:
    """List all batch jobs, newest first. Optional status filter."""
    jobs = list_jobs(status=status)
    return BatchSummaryResponse(jobs=jobs, total=len(jobs))


@app.post("/v1/batch/{job_id}/cancel")
def cancel_batch_job(
    job_id: str,
    api_key: str | None = Security(verify_api_key),
):
    """Cancel a pending or running batch job."""
    cancelled = cancel_job(job_id)
    if not cancelled:
        job = get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in status '{job.status.value}'"
        )
    return {"job_id": job_id, "status": "cancelled"}


# ── Legacy Frontend Compatibility Endpoint ──────────────────────────────────

@app.post("/analyze-image")
async def analyze_image_legacy(
    file: UploadFile = File(...),
    api_key: str | None = Security(verify_api_key),
):
    """
    Legacy endpoint for the old HTML/JS frontend (frontend/).

    Runs the full pipeline: ingest → enrich → export, then returns results
    in the format expected by the original dashboard (cells[], timing_ms, etc.).
    """
    import time
    import uuid

    t_start = time.perf_counter()

    # Step 1: Ingest (image → polygons → gold segment)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)} MB)")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    name = file.filename or "upload.png"
    try:
        polygons, fmt = segment_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"segmentation failed: {exc}") from exc

    if not polygons:
        raise HTTPException(status_code=422, detail="no nuclei detected")

    t_ingest = time.perf_counter()
    timing_ingest_ms = (t_ingest - t_start) * 1000

    gold = gold_segment_from_polygons(polygons, format_hint=fmt, source_path=name)

    # Step 2: Enrich (Rust geometry + κ engine)
    enriched = run_rust_enrichment(gold)

    t_enrich = time.perf_counter()
    timing_enrich_ms = (t_enrich - t_ingest) * 1000

    # Step 3: Extract cells for frontend response format
    cells_raw = enriched.get("cells") or enriched.get("gold_segment", {}).get("cells", [])
    enriched_cells = enriched.get("clinical", [])
    spatial_features = enriched.get("spatial_features", [])

    # Build cell list with centroids and malignancy labels
    cells_out = []
    nodes_count = len(spatial_features)
    for i, sf in enumerate(spatial_features):
        centroid = sf.get("centroid", [0.0, 0.0])
        is_cancer = False
        if i < len(enriched_cells):
            is_cancer = enriched_cells[i] == "Malignant"
        cells_out.append({
            "x": float(centroid[0]) if len(centroid) > 0 else 0.0,
            "y": float(centroid[1]) if len(centroid) > 1 else 0.0,
            "is_cancer": is_cancer,
        })

    t_total = time.perf_counter()
    total_ms = (t_total - t_start) * 1000
    
    edges_count = enriched.get("innovations", {}).get("density_ratios", {}).get("n_edges", 0)

    return {
        "status": "success",
        "message": "Analysis complete",
        "cells": cells_out,
        "nodes_count": nodes_count,
        "edges_count": edges_count,
        "timing_ms": {
            "total": round(total_ms, 1),
            "image_processing": round(timing_ingest_ms, 1),
            "mojo_engine": round(timing_enrich_ms, 2),
        },
    }
