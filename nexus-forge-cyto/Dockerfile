# syntax=docker/dockerfile:1.6
# Nexus-Forge cytology hybrid pipeline — lightweight runtime image using prebuilt binary.
# Binary is built by CI's `build` job and injected into the context.

# ---------------------------------------------------------------------------
# Stage 1 — Dependencies: Pixi (Python runtime)
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/root/.pixi/bin:${PATH}" \
    NEXUS_REPO_ROOT=/app \
    NEXUS_POLL_MS=500 \
    PYTHONPATH=/app/services

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

COPY pixi.toml /app/pixi.toml
RUN pixi install

# ---------------------------------------------------------------------------
# Prebuilt Rust binary (from CI `build` job — placed in bin/ by workflow)
# ---------------------------------------------------------------------------
COPY bin/nexus-forge-cyto-batch /app/target/release/nexus-forge-cyto-batch

# ---------------------------------------------------------------------------
# Python services, schemas, models
# ---------------------------------------------------------------------------
COPY services/ /app/services/
COPY schemas/ /app/schemas/
COPY models/ /app/models/

RUN mkdir -p /app/tmp/production_output && \
    chmod +x /app/target/release/nexus-forge-cyto-batch

EXPOSE 8501 8810

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

# Default: Streamlit dashboard. Override for API: pixi run api
ENTRYPOINT ["pixi", "run", "streamlit", "run", "services/interface_service/dashboard_app.py", \
    "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]