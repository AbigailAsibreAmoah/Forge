# ── Forge Backend — Production Dockerfile ──
FROM python:3.11-slim

# Don't write .pyc files, don't buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Railway injects $PORT at runtime; default to 8000 for local docker run
ENV PORT=8000
EXPOSE 8000

# Start uvicorn — bind to Railway's $PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
