from typing import List, Optional
import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.text_classification_service import classify_text

logger = logging.getLogger(__name__)


# ---------- Request schema ----------

class TextClassificationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Input text to classify (cannot be empty).",
    )


# ---------- Response schemas ----------

class TopKItem(BaseModel):
    label: str
    score: float


class TextClassificationResponse(BaseModel):
    label: str
    confidence: float
    top_k: List[TopKItem] = Field(default_factory=list)
    note: Optional[str] = None


# ---------- Router ----------

router = APIRouter(
    prefix="/text-classification",
    tags=["Text Classification"],
)


@router.post("/predict", response_model=TextClassificationResponse)
def classify_text_endpoint(
    payload: TextClassificationRequest,
) -> TextClassificationResponse:
    logger.info(
        "Text classification request received (chars=%d)",
        len(payload.text),
    )

    result = classify_text(payload.text)

    logger.info(
        "Text classification completed (label=%s, confidence=%.3f)",
        result.get("label", ""),
        float(result.get("confidence", 0.0)),
    )

    return TextClassificationResponse(**result)
