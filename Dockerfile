FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    LOG_LEVEL=info \
    TIMEOUT_KEEP_ALIVE=30 \
    LOCI_EXTRACT_MAX_UPLOAD_BYTES=26214400 \
    LOCI_EXTRACT_MAX_PDF_PAGES=200

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ghostscript \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()"

USER appuser

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
