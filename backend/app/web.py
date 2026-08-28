"""Static Angular delivery for single-container deployments."""

from pathlib import PurePosixPath
from typing import ClassVar

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class AngularStaticFiles(StaticFiles):
    """Serve built assets and fall back to index.html for Angular routes."""

    _reserved_roots: ClassVar[set[str]] = {
        "api",
        "audio",
        "docs",
        "experiment-audio",
        "health",
        "openapi.json",
        "redoc",
    }

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            normalized_path = str(scope.get("path", path)).lstrip("/")
            parsed_path = PurePosixPath(normalized_path)
            root = parsed_path.parts[0] if parsed_path.parts else ""
            has_file_extension = bool(parsed_path.suffix)
            if (
                error.status_code != 404
                or root in self._reserved_roots
                or has_file_extension
            ):
                raise
            return await super().get_response("index.html", scope)
