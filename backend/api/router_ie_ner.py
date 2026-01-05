from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.api.schemas import ApiResponse, Meta
from backend.core.config import settings
from backend.services.ie_ner_service import extract_entities

router = APIRouter()


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1)
    labels: Optional[List[str]] = None


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: Optional[float] = None


class NERResponse(BaseModel):
    entities: List[Entity]
    note: str

@router.post("/extract", response_model=ApiResponse[NERResponse])
def ner_extract(request: Request, req: NERRequest) -> ApiResponse[NERResponse]:
    entity_types = req.labels or ["PERSON", "ORG", "LOCATION"]

    out = extract_entities(req.text, entity_types)
    entities = []
    note = ""
    mode = "ner"
    model = None

    if isinstance(out, dict):
        entities = out.get("entities", [])
        note = out.get("note", "")
        mode = out.get("mode", "ner")
        model = out.get("model")
    elif isinstance(out, tuple):

        if len(out) >= 1:
            entities = out[0] or []
        if len(out) >= 2 and isinstance(out[1], str):
            note = out[1]
        if len(out) >= 3 and isinstance(out[2], dict):
            mode = out[2].get("mode", mode)
            model = out[2].get("model", model)
    else:
        note = f"Unexpected NER service return type: {type(out)}"

    data = NERResponse(entities=entities, note=note)

    meta = Meta(
        request_id=request.state.request_id,
        version=getattr(settings, "API_VERSION", getattr(settings, "app_version", "0.1.0")),
        mode=mode,
        model=model,
    )

    return ApiResponse(data=data, meta=meta)
