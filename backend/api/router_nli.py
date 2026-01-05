from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.nli_service import nli_predict_openai

router = APIRouter(prefix="/nli", tags=["NLI"])


class NLIRequest(BaseModel):
    premise: str = Field(..., min_length=1)
    hypothesis: str = Field(..., min_length=1)


class NLIResponse(BaseModel):
    label: str
    confidence: float
    rationale: str
    note: str


@router.post("/predict", response_model=NLIResponse)
def predict(req: NLIRequest) -> NLIResponse:
    out = nli_predict_openai(req.premise, req.hypothesis)
    return NLIResponse(**out)
