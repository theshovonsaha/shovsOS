# Backend Dockerfile
# Multi-stage build for production

FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies (gcc for compiling Python packages, libsndfile1 for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir trafilatura lxml_html_clean FlagEmbedding slowapi cryptography

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)" || exit 1

# Run the application. Override with APP_MODULE=open_claw:app for single-agent mode.
CMD ["sh", "-c", "uvicorn ${APP_MODULE:-api.main:app} --host 0.0.0.0 --port ${PORT:-8000}"]
