from __future__ import annotations

import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from backend.api.schemas import ApiResponse, Meta
from backend.core.config import settings
from backend.services.translation_service import translate_text_tagged

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/translation",
    tags=["Machine Translation"],
)


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to translate (cannot be empty).")
    source_lang: str = Field(..., min_length=2, max_length=12, description="Source language code (e.g., en).")
    target_lang: str = Field(..., min_length=2, max_length=12, description="Target language code (e.g., fr).")


class TranslationResponse(BaseModel):
    translated_text: str
    note: str


@router.post("/translate", response_model=ApiResponse[TranslationResponse])
def translate(request: Request, payload: TranslationRequest) -> ApiResponse[TranslationResponse]:
    logger.info(
        "Translation request received (chars=%d, %s->%s)",
        len(payload.text),
        payload.source_lang,
        payload.target_lang,
    )

    translated_text, note = translate_text_tagged(
        payload.text,
        payload.source_lang,
        payload.target_lang,
    )

    logger.info("Translation completed (output_chars=%d)", len(translated_text))

    data = TranslationResponse(translated_text=translated_text, note=note)

    meta = Meta(
        request_id=request.state.request_id,
        version=getattr(settings, "API_VERSION", getattr(settings, "app_version", "0.1.0")),
        mode=("llm" if settings.use_llm_translation else "dummy"),
        model=(settings.openai_model if settings.use_llm_translation else None),
    )

    return ApiResponse(data=data, meta=meta)
