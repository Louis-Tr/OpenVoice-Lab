"""HTTP mappings for backend domain errors."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.audio.service import AudioStorageError
from app.inference.base import InferenceError, UnsupportedVoiceError
from app.metrics.collector import MetricsCollectionError
from app.models.loader import ModelLoadError
from app.models.registry import ModelNotFoundError


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

    @application.exception_handler(UnsupportedVoiceError)
    async def unsupported_voice(
        _request: Request,
        error: UnsupportedVoiceError,
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
