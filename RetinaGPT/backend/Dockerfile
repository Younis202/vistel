# ============================================================
# Retina-GPT Production Dockerfile
# Multi-stage build for lean production image
# ============================================================

# ── Stage 1: Build dependencies ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: Production image ────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="Retina-GPT Engineering Team"
LABEL description="Retina-GPT: AI-Powered Retinal Analysis API"
LABEL version="1.0.0"

# Runtime dependencies (OpenCV needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy project source
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 retinagpt && \
    chown -R retinagpt:retinagpt /app
USER retinagpt

# Make sure local packages are on PATH
ENV PATH=/home/retinagpt/.local/bin:/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# API settings
ENV RETINA_GPT_CHECKPOINT=/app/checkpoints/checkpoint_best.pth
ENV API_KEYS=change-me-in-production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
