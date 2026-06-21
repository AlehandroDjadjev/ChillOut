FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5173 \
    CLOUD_MODEL_PORT=7860 \
    CLOUD_MODEL_HOST=127.0.0.1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt package.json ./
COPY functions/api/package.json functions/api/package-lock.json ./functions/api/

RUN pip install --no-cache-dir -r requirements.txt \
    && npm --prefix functions/api ci --omit=dev

COPY . .

EXPOSE 5173

CMD ["bash", "scripts/run_hosted_cloud_model.sh"]
