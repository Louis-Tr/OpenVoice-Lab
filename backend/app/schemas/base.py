"""Shared behavior for public API schemas."""

from pydantic import BaseModel, ConfigDict


def to_camel(value: str) -> str:
    """Convert internal snake_case field names to JSON camelCase."""
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiSchema(BaseModel):
    """Base contract with consistent JSON aliases."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
