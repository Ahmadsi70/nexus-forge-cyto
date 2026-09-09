# Deployment Guide

This guide covers deploying Nexus-Forge Cyto in production environments, from
single-server Docker to multi-node setups.

## Deployment Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Docker Compose (single server)** | All services on one machine | Small labs, evaluation, development |
| **Docker Swarm / Kubernetes** | Multi-node with orchestration | High-volume labs, Pharma CRO |
| **Bare metal CLI** | Rust binary + Python venv | Edge deployment, air-gapped systems |
| **RunPod / Cloud GPU** | GPU-accelerated SAM 3 inference | Cloud-based analysis |

---

## Docker Compose (Recommended)

### Prerequisites

- Docker >= 24.0
- Docker Compose >= 2.20
- 8+ GB RAM, 4+ CPU cores
- 10+ GB free disk (for models and output)

### Quick Deploy

```bash
cd nexus-forge-cyto

# Copy and customize environment
cp .env.example .env
# Edit .env — set NEXUS_API_KEY for authentication

# Start services
docker-compose up -d

# Verify
curl http://localhost:8810/ready
# → {"status": "ready", "checks": {...}}

curl http://localhost:8501/_stcore/health
# → OK
```

### Services

| Service | Port | Purpose | Health Check |
|---------|------|---------|-------------|
| nexus-forge | 8501 | Streamlit dashboard | `/_stcore/health` |
| nexus-forge-api | 8810 | FastAPI REST API | `/ready` |

### Resource Tuning

Override defaults via environment variables:

```bash
# Increase memory for large slides
NEXUS_DOCKER_MEM=16g \
NEXUS_DOCKER_MEM_RESERVE=8g \
NEXUS_DOCKER_CPUS=8.0 \
docker-compose up -d
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEXUS_DOCKER_MEM` | `8g` | Hard memory limit per container |
| `NEXUS_DOCKER_MEM_RESERVE` | `4g` | Soft memory reservation |
| `NEXUS_DOCKER_MEMSWAP` | `12g` | Memory + swap limit |
| `NEXUS_DOCKER_CPUS` | `4.0` | CPU limit |

### Persistent Storage

The named volume `nexus_production_output` persists analysis results across
container restarts. Mount an external path for production:

```yaml
volumes:
  - /data/nexus-output:/app/tmp/production_output
```

---

## Kubernetes (Basic)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-forge-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nexus-forge-api
  template:
    metadata:
      labels:
        app: nexus-forge-api
    spec:
      containers:
      - name: api
        image: nexus-forge-cyto:latest
        command: ["pixi", "run", "api"]
        ports:
        - containerPort: 8810
        env:
        - name: NEXUS_RUST_BRIDGE
          value: "subprocess"
        - name: NEXUS_API_KEY
          valueFrom:
            secretKeyRef:
              name: nexus-secrets
              key: api-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        readinessProbe:
          httpGet:
            path: /ready
            port: 8810
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /live
            port: 8810
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-forge-api
spec:
  selector:
    app: nexus-forge-api
  ports:
  - port: 8810
    targetPort: 8810
```

---

## Security Checklist

### Production Hardening

- [ ] Set `NEXUS_API_KEY` to a strong random value (64+ chars)
- [ ] Restrict `NEXUS_ALLOWED_ORIGINS` to your frontend domain(s)
- [ ] Place behind a reverse proxy (nginx/Traefik) with TLS termination
- [ ] Enable HTTPS — add TLS certificates to the reverse proxy
- [ ] Set rate limits: `NEXUS_RATE_LIMIT` and `NEXUS_RATE_WINDOW`
- [ ] Run containers as non-root user (add `USER nexus` to Dockerfile)
- [ ] Use Docker secrets or Kubernetes secrets for API keys (never hardcode)
- [ ] Enable audit logging by setting `NEXUS_LOG=info`
- [ ] Restrict network access: only expose port 8810 to internal network

### Reverse Proxy (nginx example)

```nginx
server {
    listen 443 ssl;
    server_name nexus.example.com;

    ssl_certificate     /etc/ssl/nexus.crt;
    ssl_certificate_key /etc/ssl/nexus.key;

    location / {
        proxy_pass http://127.0.0.1:8810;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Rate limiting
        limit_req zone=nexus burst=20 nodelay;
        limit_req_status 429;

        # Max upload size
        client_max_body_size 100m;
    }
}

limit_req_zone $binary_remote_addr zone=nexus:10m rate=100r/m;
```

---

## Monitoring

### Health Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /live` | Liveness — is process alive? | `{"status": "ok"}` |
| `GET /ready` | Readiness — can we serve traffic? | `{"status": "ready", "checks": {...}}` |
| `GET /health` | Legacy health check | `{"status": "ok"}` |

### Logging

Set `NEXUS_LOG=info` (or `debug` for verbose output) to control log verbosity.
Logs are written to stderr with structured format:

```
2026-09-01T12:34:56 [INFO ] [nexus.api] [req_id=a1b2c3] POST /v1/enrich -> 200 (45.2ms) [ip=10.0.0.5]
```

In Docker, view logs with:
```bash
docker-compose logs -f nexus-forge-api
```

### Resource Monitoring

- **CPU**: Monitor `nexus-forge-cyto-batch` process — it runs Rayon parallel
  computation across all cores
- **Memory**: Peak usage is ~2× the largest slide's cell count in bytes.
  Large slides (10K+ cells) need 4+ GB
- **Disk**: Output JSON files are ~1 KB per cell. 10K cells ≈ 10 MB output.
  Implement log rotation for `tmp/production_output/`

---

## Backup & Recovery

### What to Back Up

- `.env` configuration file
- `models/` directory (trained model weights)
- `tmp/production_output/` (analysis results — if needed)
- Custom trained models from `training/checkpoints/`

### Recovery

```bash
# After hardware failure
git clone https://github.com/clinicalguard/nexus-forge-cyto.git
cp backup/.env cancer_project/
cp -r backup/models/* cancer_project/models/
docker-compose up -d
```

---

## Air-Gapped Deployment

For hospital environments without internet access:

1. Build on an internet-connected machine:
   ```bash
   docker build -t nexus-forge-cyto:latest .
   docker save nexus-forge-cyto:latest | gzip > nexus-forge-cyto.tar.gz
   ```

2. Transfer the tarball and model files to the air-gapped machine:
   ```bash
   scp nexus-forge-cyto.tar.gz user@airgap-host:/tmp/
   scp models/*.onnx models/*.joblib user@airgap-host:/path/to/models/
   ```

3. Load and run:
   ```bash
   docker load < nexus-forge-cyto.tar.gz
   docker-compose up -d
   ```

The `offline/wheels/` directory contains pre-downloaded Python wheels for
offline pip install.