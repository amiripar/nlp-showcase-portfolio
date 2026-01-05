from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.nlu_service import nlu_parse_openai

router = APIRouter(prefix="/nlu", tags=["NLU"])


class NLURequest(BaseModel):
    text: str = Field(..., min_length=1, description="User input text to parse (intent/entities/slots).")


class IntentOut(BaseModel):
    name: str
    confidence: float


class EntityOut(BaseModel):
    type: str
    text: str
    start: int
    end: int
    value: Any


class NLUResponse(BaseModel):
    intent: IntentOut
    entities: List[EntityOut]
    slots: Dict[str, Any]
    needs_clarification: bool
    clarifying_question: str
    note: str


@router.post("/parse", response_model=NLUResponse)
def parse(req: NLURequest) -> NLUResponse:
    """
    Parse user text into:
    - intent (name + confidence)
    - entities (type, text, offsets, normalized value)
    - slots (dict of arguments)
    - clarification flags (if needed)
    """
    out = nlu_parse_openai(req.text)
    return NLUResponse(**out)
