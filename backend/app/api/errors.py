"""HTTP mappings for backend domain errors."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.audio.service import AudioStorageError
from app.benchmark.service import BenchmarkJobNotFoundError
from app.experiments.common import (
    ExperimentEvidenceError,
    ExperimentFixtureNotFoundError,
    ExperimentJobNotFoundError,
    ExperimentModelNotFoundError,
    ExperimentQueueFullError,
)
from app.inference.base import InferenceError, UnsupportedVoiceError
from app.metrics.collector import MetricsCollectionError
from app.models.loader import ModelLoadError
from app.models.registry import ModelNotFoundError
from app.text_processing.service import TextProcessingError


def register_error_handlers(application: FastAPI) -> None:
    """Keep transport error mapping out of feature controllers."""

    @application.exception_handler(ModelNotFoundError)
    async def model_not_found(
        _request: Request,
        error: ModelNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(error)},
        )

    @application.exception_handler(BenchmarkJobNotFoundError)
    @application.exception_handler(ExperimentFixtureNotFoundError)
    @application.exception_handler(ExperimentJobNotFoundError)
    @application.exception_handler(ExperimentModelNotFoundError)
    async def benchmark_job_not_found(
        _request: Request,
        error: (
            BenchmarkJobNotFoundError
            | ExperimentFixtureNotFoundError
            | ExperimentJobNotFoundError
            | ExperimentModelNotFoundError
        ),
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(error)},
        )

    @application.exception_handler(ExperimentQueueFullError)
    async def experiment_queue_full(
        _request: Request,
        error: ExperimentQueueFullError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": str(error)},
        )

    @application.exception_handler(ExperimentEvidenceError)
    async def experiment_unavailable(
        _request: Request,
        error: ExperimentEvidenceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(error)},
        )

    @application.exception_handler(UnsupportedVoiceError)
    @application.exception_handler(TextProcessingError)
    async def invalid_synthesis_input(
        _request: Request,
        error: UnsupportedVoiceError | TextProcessingError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(error)},
        )

    @application.exception_handler(ModelLoadError)
    async def model_unavailable(
        _request: Request,
        error: ModelLoadError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(error)},
        )

    @application.exception_handler(InferenceError)
    @application.exception_handler(AudioStorageError)
    @application.exception_handler(MetricsCollectionError)
    async def synthesis_failed(
        _request: Request,
        error: InferenceError | AudioStorageError | MetricsCollectionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(error)},
        )
