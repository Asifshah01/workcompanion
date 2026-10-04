# =============================================================================
# WorkCompanion AI
# =============================================================================
# Build:  docker build -t workcompanion .
# Run:    docker run -p 8501:8501 --env-file .env -v wcdata:/app/data workcompanion
#
# The image is built without baked-in secrets: GROQ_API_KEY is supplied at run
# time. Without it the app still starts, on its deterministic offline provider.
# =============================================================================

FROM python:3.12-slim AS base

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep BLAS single-threaded; torch spawns its own pool and containers
    # usually only get a couple of cores.
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    # Hugging Face is only reached once, for the embedding model.
    HF_HOME=/app/.cache/huggingface \
    DATA_DIR=/app/data

# ---------------------------------------------------------------------------
# System packages
#
# libgl1/libglib2.0 are required by chromadb's onnxruntime dependency, which
# sentence-transformers pulls in.  tesseract is for OCR fallback on scanned
# PDFs; remove it (and set OCR_ENGINE=none) if you do not need it - it adds
# ~15 MB.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        libglib2.0-0 \
        libgl1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies, in their own layer so code edits do not reinstall torch.
# ---------------------------------------------------------------------------
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
COPY pyproject.toml ./
COPY app.py ./
COPY workcompanion/ ./workcompanion/
COPY scripts/ ./scripts/
COPY demo/ ./demo/
COPY .env.example ./

# Install the package itself, without re-resolving dependencies.
RUN python -m pip install --no-deps -e .

# Pre-download the embedding model at build time so the first request is fast
# and the container can start with no outbound network access.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}')" || \
    echo 'warning: embedding model could not be pre-fetched; it will download on first use'

# Run as a non-root user; /app/data is the only writable location it needs.
RUN useradd --create-home --uid 10001 wc \
    && mkdir -p /app/data /app/.cache \
    && chown -R wc:wc /app/data /app/.cache
USER wc

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]