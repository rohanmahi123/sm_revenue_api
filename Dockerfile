# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install into a separate folder so we can copy only that
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Final slim image ─────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy only app code
COPY auth.py config.py database.py main.py models.py schemas.py security.py ./
COPY db/          ./db/
COPY ml/          ./ml/
COPY routers/     ./routers/
COPY model_store/ ./model_store/
COPY sm_revenue.db ./

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
