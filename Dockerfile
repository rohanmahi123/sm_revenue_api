# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# gcc + libpq-dev  → psycopg2
# libgomp1         → scikit-learn (OpenMP for parallel jobs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer cached until requirements change) ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY auth.py config.py database.py main.py models.py schemas.py security.py ./
COPY db/       ./db/
COPY ml/       ./ml/
COPY routers/  ./routers/

# ── Create model store directory ──────────────────────────────────────────────
RUN mkdir -p model_store

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
