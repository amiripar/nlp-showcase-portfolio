from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from backend.services.qa_service import answer_question_grounded
from backend.api.schemas import ApiResponse, Meta
from backend.core.config import settings

router = APIRouter()


class QARequest(BaseModel):
    context: str = Field(..., min_length=1, description="Reference text to answer from.")
    question: str = Field(..., min_length=1, description="User question.")


class QAResponse(BaseModel):
    answer: str
    evidence: str
    note: str


@router.post("/answer", response_model=ApiResponse[QAResponse])
def qa_answer(request: Request, req: QARequest) -> ApiResponse[QAResponse]:
    answer, evidence, note = answer_question_grounded(req.context, req.question)

    data = QAResponse(answer=answer, evidence=evidence, note=note)

    meta = Meta(
        request_id=request.state.request_id,
        version=getattr(settings, "API_VERSION", getattr(settings, "app_version", "0.1.0")),
        mode="grounded",
        model=None,  
        )

    return ApiResponse(data=data, meta=meta)
