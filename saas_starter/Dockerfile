# saas-starter -- Milestone 2 deploy config
#
# Build context is this directory (saas_starter/), e.g.:
#   cd saas_starter && docker build -t saas-starter .
# or from the repo root:
#   docker build -t saas-starter -f saas_starter/Dockerfile saas_starter

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer only rebuilds when
# requirements.txt actually changes, not on every source-code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user -- the image doesn't need root for anything.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

# SQLite lives on a mounted volume (see docker-compose.yml) so data
# survives container recreation. Never bake SAAS_JWT_SECRET into the
# image -- it's read from the environment at runtime, with the same
# obviously-fake dev default the app itself falls back to (see
# app/__init__.py) if the operator forgets to set it.
ENV SAAS_DB_PATH=/data/saas_starter.db

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=2).status == 200 else 1)"

# 2 workers is a reasonable default for a small starter deployment; tune
# via `docker run ... gunicorn --workers N ...` override for real traffic.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:create_app()"]
