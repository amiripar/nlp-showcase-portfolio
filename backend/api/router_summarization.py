import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.summarization_service import summarize_text_simple

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/summarization",
    tags=["Summarization"],
)


class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to summarize (cannot be empty).")
    max_chars: int = Field(120, ge=30, le=3000, description="Max characters for dummy summary (30..3000).")


class SummarizationResponse(BaseModel):
    summary: str
    compression_ratio: float


@router.post("/summarize", response_model=SummarizationResponse)
def summarize(payload: SummarizationRequest) -> SummarizationResponse:
    logger.info("Summarization request received (chars=%d, max_chars=%d)", len(payload.text), payload.max_chars)

    summary, ratio = summarize_text_simple(payload.text, payload.max_chars)

    logger.info("Summarization completed (summary_chars=%d, ratio=%.3f)", len(summary), ratio)

    return SummarizationResponse(summary=summary, compression_ratio=ratio)
