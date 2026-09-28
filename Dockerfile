# Temkinli hisse tarama — tek imaj; compose'daki scheduler ve dashboard servisleri ortak kullanır.
FROM python:3.13-slim

# ARROW_DEFAULT_MEMORY_POOL=system: Arrow's own allocator keeps freed parquet buffers; the system one
# returns them (measured ~50 MB lower peak on the daily flow)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARROW_DEFAULT_MEMORY_POOL=system \
    TZ=Europe/Istanbul

# libgomp1: LightGBM; tzdata: zamanlayıcı saat dilimi
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements.lock /app/deploy/requirements.lock
RUN pip install -r deploy/requirements.lock

COPY . /app

# kalıcı veri (compose'da volume olarak bağlanır)
VOLUME ["/app/storage", "/app/model/registry"]

CMD ["python", "main.py", "schedule"]
