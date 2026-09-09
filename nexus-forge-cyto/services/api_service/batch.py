"""
Batch async processing API for high-throughput slide analysis.

Why: pharma trials and clinical labs process hundreds of slides per day.
A synchronous pipeline would timeout or block other requests.

Architecture:
  - Submit: POST /v1/batch with list of image paths or annotation files
  - Status:  GET /v1/batch/{job_id} for progress polling
  - Results: Downloadable JSON when complete
  - Webhook: POST to callback URL on completion (optional)

The batch engine uses asyncio background tasks (no external queue dependency)
with configurable worker count via NEXUS_BATCH_WORKERS env var.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from api_service.pipeline_core import (
    PipelineError,
    gold_segment_from_vector_bytes,
    run_rust_enrichment_legacy as run_rust_enrichment,
)

# ── Job model ─────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Single item in a batch job."""

    item_id: int
    filename: str
    status: JobStatus = JobStatus.PENDING
    cell_count: int = 0
    error: str | None = None
    result: dict[str, Any] | None = None
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class BatchJob:
    """A batch processing job with multiple items."""

    job_id: str
    items: list[BatchItem]
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    completed: int = 0
    failed: int = 0
    total: int = 0
    webhook_url: str | None = None
    error: str | None = None

    def __post_init__(self):
        self.total = len(self.items)

    def progress_pct(self) -> float:
        if self.total == 0:
            return 0.0
        done = self.completed + self.failed
        return done / self.total

    def to_summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "progress_pct": round(self.progress_pct() * 100, 1),
            "error": self.error,
        }

    def to_detail(self) -> dict[str, Any]:
        detail = self.to_summary()
        detail["items"] = [
            {
                "item_id": item.item_id,
                "filename": item.filename,
                "status": item.status.value,
                "cell_count": item.cell_count,
                "error": item.error,
                "duration_ms": round(
                    ((item.finished_at or 0) - (item.started_at or 0)) * 1000, 1
                ) if item.finished_at else None,
            }
            for item in self.items
        ]
        return detail


# ── In-memory job store ───────────────────────────────────────────────────────

# In production, replace with Redis or a database.
# The current in-memory store is sufficient for single-server deployments
# and is reset on process restart.
_jobs: dict[str, BatchJob] = {}

# Maximum job age in seconds before cleanup (default: 24 hours)
_JOB_TTL_SECONDS = int(os.environ.get("NEXUS_BATCH_JOB_TTL", str(24 * 3600)))

# Maximum concurrent workers (default: 2)
_MAX_WORKERS = int(os.environ.get("NEXUS_BATCH_WORKERS", "2"))


def _cleanup_expired_jobs() -> int:
    """Remove jobs older than TTL. Returns number of cleaned jobs."""
    cutoff = time.time() - _JOB_TTL_SECONDS
    expired = []
    for job_id, job in list(_jobs.items()):
        try:
            created = datetime.fromisoformat(job.created_at)
            if created.timestamp() < cutoff:
                expired.append(job_id)
        except (ValueError, TypeError):
            expired.append(job_id)

    for job_id in expired:
        del _jobs[job_id]
    return len(expired)


def create_job(
    filenames: list[str],
    file_bytes_list: list[bytes],
    webhook_url: str | None = None,
) -> BatchJob:
    """
    Create a new batch job from a list of (filename, bytes) pairs.

    Args:
        filenames: Original filenames for each upload
        file_bytes_list: Raw bytes for each file
        webhook_url: Optional URL to POST results to on completion

    Returns:
        The created BatchJob
    """
    _cleanup_expired_jobs()

    job_id = f"batch_{uuid.uuid4().hex[:12]}"
    items = []
    for idx, (fname, raw) in enumerate(zip(filenames, file_bytes_list)):
        items.append(BatchItem(item_id=idx, filename=fname))

    job = BatchJob(
        job_id=job_id,
        items=items,
        webhook_url=webhook_url,
    )
    _jobs[job_id] = job

    # Store raw bytes separately (not in job state, to keep summary lean)
    _raw_store[job_id] = file_bytes_list

    # Start processing in background
    asyncio.create_task(_process_job(job_id))

    return job


# Raw file bytes store (separate from job metadata)
_raw_store: dict[str, list[bytes]] = {}


def get_job(job_id: str) -> BatchJob | None:
    return _jobs.get(job_id)


def cancel_job(job_id: str) -> bool:
    """Cancel a running or pending job. Returns True if cancelled."""
    _cleanup_expired_jobs()
    job = _jobs.get(job_id)
    if job is None:
        return False
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return False
    job.status = JobStatus.CANCELLED
    job.finished_at = datetime.now(timezone.utc).isoformat()
    return True


def list_jobs(status: str | None = None) -> list[dict[str, Any]]:
    """List all jobs, optionally filtered by status."""
    _cleanup_expired_jobs()
    result = []
    for job in _jobs.values():
        if status and job.status.value != status:
            continue
        result.append(job.to_summary())
    # Newest first
    result.sort(key=lambda j: j["created_at"], reverse=True)
    return result


# ── Background processing ─────────────────────────────────────────────────────


async def _process_job(job_id: str) -> None:
    """Background task: process each item in a batch job with bounded concurrency."""
    job = _jobs.get(job_id)
    if job is None:
        return

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc).isoformat()

    raw_list = _raw_store.pop(job_id, [])
    if len(raw_list) != len(job.items):
        job.status = JobStatus.FAILED
        job.error = f"Raw data mismatch: {len(raw_list)} bytes vs {len(job.items)} items"
        job.finished_at = datetime.now(timezone.utc).isoformat()
        return

    # Process with semaphore-limited concurrency
    semaphore = asyncio.Semaphore(_MAX_WORKERS)

    async def process_item(item: BatchItem, raw: bytes):
        async with semaphore:
            item.status = JobStatus.RUNNING
            item.started_at = time.time()
            try:
                # Offload CPU-bound work to thread pool
                result = await asyncio.to_thread(_process_single, item.filename, raw)
                item.status = JobStatus.COMPLETED
                item.cell_count = result.get("cell_count", 0)
                item.result = result
                job.completed += 1
            except Exception as exc:
                item.status = JobStatus.FAILED
                item.error = f"{type(exc).__name__}: {exc}"
                item.cell_count = 0
                job.failed += 1
            finally:
                item.finished_at = time.time()

    # Process all items
    tasks = [
        process_item(item, raw)
        for item, raw in zip(job.items, raw_list)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Finalize
    if job.failed == job.total:
        job.status = JobStatus.FAILED
        job.error = f"All {job.total} items failed"
    else:
        job.status = JobStatus.COMPLETED

    job.finished_at = datetime.now(timezone.utc).isoformat()

    # Fire webhook if configured
    if job.webhook_url:
        await _fire_webhook(job)


def _process_single(filename: str, raw: bytes) -> dict[str, Any]:
    """
    Synchronous processing of a single file through the full pipeline.

    This runs in a thread pool via asyncio.to_thread().
    """
    t_start = time.perf_counter()

    # Step 1: Ingest
    gold = gold_segment_from_vector_bytes(filename, raw)
    t_ingest = time.perf_counter()

    # Step 2: Enrich (Rust geometry + kappa engine)
    enriched = run_rust_enrichment(gold)
    t_enrich = time.perf_counter()

    # Extract summary
    cell_count = int(enriched.get("cell_count", 0))
    clinical = enriched.get("clinical", [])
    malignant = sum(1 for c in clinical if str(c).strip().lower() == "malignant")
    immune = sum(1 for c in clinical if str(c).strip().lower() == "immune")
    normal = cell_count - malignant - immune

    return {
        "filename": filename,
        "cell_count": cell_count,
        "malignant_count": malignant,
        "immune_count": immune,
        "normal_count": normal,
        "timing_ms": {
            "ingest": round((t_ingest - t_start) * 1000, 1),
            "enrich": round((t_enrich - t_ingest) * 1000, 1),
            "total": round((time.perf_counter() - t_start) * 1000, 1),
        },
        "enriched": enriched,  # Full enriched output
    }


async def _fire_webhook(job: BatchJob) -> None:
    """POST job results to the configured webhook URL."""
    try:
        import aiohttp

        payload = job.to_detail()
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                job.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Nexus-Forge-Batch/0.2"},
            ) as resp:
                if resp.status >= 400:
                    print(f"[WARN] Webhook {job.webhook_url} returned {resp.status}")
    except ImportError:
        # aiohttp not available — try requests in thread
        try:
            import requests

            await asyncio.to_thread(
                requests.post,
                job.webhook_url,
                json=job.to_detail(),
                headers={"Content-Type": "application/json", "User-Agent": "Nexus-Forge-Batch/0.2"},
                timeout=30,
            )
        except Exception as exc:
            print(f"[WARN] Webhook {job.webhook_url} failed: {exc}")
    except Exception as exc:
        print(f"[WARN] Webhook {job.webhook_url} failed: {exc}")


# ── Batch submission helpers for API ──────────────────────────────────────────


def validate_batch_files(
    files: list[tuple[str, bytes]],
    max_files: int = 100,
    max_total_size: int = 500 * 1024 * 1024,  # 500 MB
) -> list[str]:
    """
    Validate batch submission files.

    Returns list of validation error messages (empty = valid).
    """
    errors = []

    if not files:
        errors.append("No files provided")
        return errors

    if len(files) > max_files:
        errors.append(f"Too many files: {len(files)} (max {max_files})")

    total_size = sum(len(raw) for _, raw in files)
    if total_size > max_total_size:
        errors.append(
            f"Total size {total_size / (1024*1024):.1f} MB exceeds "
            f"max {max_total_size / (1024*1024):.0f} MB"
        )

    for fname, raw in files:
        if not raw:
            errors.append(f"'{fname}' is empty")
        if not fname:
            errors.append("Empty filename")

    return errors