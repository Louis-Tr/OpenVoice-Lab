FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 openvoice \
    && useradd --system --uid 10001 --gid openvoice --home-dir /app openvoice

WORKDIR /app

COPY backend/pyproject.toml ./pyproject.toml
COPY backend/requirements.lock ./requirements.lock
COPY backend/app ./app

RUN python -m pip install --requirement requirements.lock \
    && python -m pip check

COPY --chown=openvoice:openvoice backend/scripts ./scripts
COPY --chown=openvoice:openvoice backend/model-artifacts/README.md ./model-artifacts/README.md

RUN mkdir -p artifacts/audio benchmark-results \
    && chown -R openvoice:openvoice /app

USER openvoice

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
