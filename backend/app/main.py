"""FastAPI application composition root."""

from fastapi import FastAPI

from app.api import benchmarks, health, models, synthesis


def create_app() -> FastAPI:
    """Create the API and compose its thin HTTP routers."""
    application = FastAPI(
        title="OpenVoice Lab API",
        version="0.1.0",
        description="Architecture scaffold; synthesis and benchmarking are not implemented.",
    )
    application.include_router(synthesis.router, prefix="/api")
    application.include_router(models.router, prefix="/api")
    application.include_router(benchmarks.router, prefix="/api")
    application.include_router(health.router)
    return application


app = create_app()

