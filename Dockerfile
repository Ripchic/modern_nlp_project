# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.5.26

# Copy dependency manifest first for better cache utilisation
COPY pyproject.toml ./
RUN touch README.md

# Create virtual environment and install all production dependencies
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python -e . --no-cache

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

# Make the venv the default Python
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src"

USER appuser

EXPOSE 8000

# Default command: run FastAPI via uvicorn
# Override in docker-compose.yml per-service with `command:`
CMD ["uvicorn", "reviewmind.main:app", "--host", "0.0.0.0", "--port", "8000"]
