# Multi-stage Dockerfile for production deployment on Render.com
# Optimized for FastAPI with OpenCASCADE, NLP libraries, and 3D processing

# Stage 1: Builder for dependencies
FROM python:3.11-slim as builder

# Build arguments for optimization
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    make \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libxmu-dev \
    libxi-dev \
    libgmp-dev \
    libmpfr-dev \
    libeigen3-dev \
    libboost-all-dev \
    libopenblas-dev \
    libhdf5-dev \
    pkg-config \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt ./

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel for faster builds
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install Python packages with optimizations
RUN pip install --no-cache-dir \
    --disable-pip-version-check \
    --compile \
    -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn[gevent] \
    uvicorn[standard] \
    prometheus-client \
    python-multipart

# Stage 2: Runtime optimized for Render.com
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa \
    libgl1-mesa-glx \
    libxmu6 \
    libxi6 \
    libgomp1 \
    libopenblas0 \
    libhdf5-103 \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set production environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    WEB_CONCURRENCY=4 \
    MAX_WORKERS=4 \
    WORKER_TIMEOUT=300 \
    KEEP_ALIVE=5 \
    PORT=10000

# Create non-root user for security
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 1000 appuser && \
    mkdir -p /app /app/static /app/logs /tmp/text-to-cad /app/uploads && \
    chown -R appuser:appgroup /app /tmp/text-to-cad

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . .

# Create necessary directories and set permissions
RUN mkdir -p /app/static/opencascade && \
    chown -R appuser:appgroup /app/static && \
    chmod -R 755 /app/static

# Copy startup scripts
COPY --chown=appuser:appgroup scripts/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Expose port (Render uses PORT environment variable)
EXPOSE $PORT

# Health check optimized for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Production startup command
CMD ["/app/start.sh"]