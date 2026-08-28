FROM node:22.18.0-alpine3.22@sha256:1b2479dd35a99687d6638f5976fd235e26c5b37e8122f786fcd5fe231d63de5b AS frontend-build

WORKDIR /workspace

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/angular.json frontend/tsconfig.json frontend/tsconfig.app.json ./
COPY frontend/src ./src
RUN npm run build


FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS model-provisioner

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY backend/scripts/download_models.py ./backend/scripts/download_models.py
COPY backend/model-artifacts/README.md ./backend/model-artifacts/README.md
RUN python backend/scripts/download_models.py


FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENVOICE_ENVIRONMENT=production \
    OPENVOICE_MODEL_ARTIFACT_DIR=/app/model-artifacts \
    OPENVOICE_GENERATED_AUDIO_DIR=/tmp/openvoice/audio \
    OPENVOICE_BENCHMARK_RESULT_DIR=/tmp/openvoice/benchmarks \
    OPENVOICE_STAGE11_ARTIFACT_ROOT=/tmp/openvoice/stage11/full-training \
    OPENVOICE_STAGE11_APPROACH_RUN_ROOT=/tmp/openvoice/stage11/agent-runs \
    OPENVOICE_STAGE11_MANIFEST_ROOT=/tmp/openvoice/stage11/manifests \
    OPENVOICE_STAGE12_ARTIFACT_ROOT=/tmp/openvoice/stage12 \
    OPENVOICE_EXPERIMENT_MODEL_CACHE_DIR=/tmp/openvoice/stage12/model-cache \
    OPENVOICE_EXPERIMENT_SPEAKER_PROFILE_DIR=/tmp/openvoice/stage12/serving-profile \
    OPENVOICE_FRONTEND_DIST_DIR=/app/frontend

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 openvoice \
    && useradd --system --uid 10001 --gid openvoice --home-dir /app openvoice

WORKDIR /app

COPY backend/pyproject.toml ./pyproject.toml
COPY backend/requirements.lock ./requirements.lock
RUN python -m pip install --requirement requirements.lock \
    && python -m pip check

COPY --chown=openvoice:openvoice backend/app ./app
COPY --chown=openvoice:openvoice --from=model-provisioner /workspace/backend/model-artifacts ./model-artifacts
COPY --chown=openvoice:openvoice --from=frontend-build /workspace/dist/openvoice-lab/browser ./frontend

USER openvoice

EXPOSE 8080

CMD ["sh", "-c", "exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
