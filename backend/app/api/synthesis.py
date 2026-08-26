"""HTTP contract for synthesis requests."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.synthesis import SynthesisRequest, SynthesisResult

router = APIRouter(tags=["synthesis"])


@router.post(
    "/synthesis",
    response_model=SynthesisResult,
    status_code=status.HTTP_200_OK,
)
async def create_synthesis(_request: SynthesisRequest) -> SynthesisResult:
    """Accept the contract while orchestration remains intentionally unimplemented."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Synthesis orchestration is not implemented yet.",
    )

