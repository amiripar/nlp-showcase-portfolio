from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Request, UploadFile, File
from pydantic import BaseModel, Field

from backend.api.schemas import ApiResponse, Meta
from backend.core.config import settings
from backend.services.rag_service import rag_index, rag_ask, rag_index_pdf, rag_clear, rag_list_corpora, rag_delete

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG (Retrieval-Augmented Generation)"])


class RagIndexRequest(BaseModel):
    documents: List[str] = Field(..., min_items=1, description="Documents (raw text) to index.")


class RagIndexResponse(BaseModel):
    stats: Dict[str, Any]
    note: str


class RagAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    corpus_id: Optional[str] = Field(None, description="Optional corpus_id. If omitted, uses active corpus.")


class RagAskResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]]
    note: str


def _meta(request: Request, mode: str) -> Meta:
    return Meta(
        request_id=getattr(request.state, "request_id", "unknown"),
        version=getattr(settings, "API_VERSION", getattr(settings, "app_version", "0.1.0")),
        mode=mode,
        model=getattr(settings, "openai_model", None),
    )


@router.get("/corpora", response_model=ApiResponse[List[Dict[str, Any]]])
def corpora(request: Request) -> ApiResponse[List[Dict[str, Any]]]:
    data = rag_list_corpora()
    return ApiResponse(data=data, meta=_meta(request, mode="rag_corpora_list"))


@router.post("/index", response_model=ApiResponse[RagIndexResponse])
def index_docs(request: Request, payload: RagIndexRequest) -> ApiResponse[RagIndexResponse]:
    logger.info("RAG index request received (docs=%d)", len(payload.documents))
    stats, note = rag_index(payload.documents)
    logger.info("RAG index completed (chunks=%s)", stats.get("chunks_indexed"))

    data = RagIndexResponse(stats=stats, note=note)
    return ApiResponse(data=data, meta=_meta(request, mode="rag_index"))


@router.post("/index_pdf", response_model=ApiResponse[RagIndexResponse])
async def index_pdf(request: Request, file: UploadFile = File(...)) -> ApiResponse[RagIndexResponse]:
    if not (file.filename or "").lower().endswith(".pdf"):
        data = RagIndexResponse(stats={"chunks_indexed": 0}, note="Please upload a .pdf file only.")
        return ApiResponse(data=data, meta=_meta(request, mode="rag_index_pdf"))

    pdf_bytes = await file.read()
    stats, note = rag_index_pdf(pdf_bytes, filename=file.filename or "uploaded.pdf")

    data = RagIndexResponse(stats=stats, note=note)
    return ApiResponse(data=data, meta=_meta(request, mode="rag_index_pdf"))


@router.post("/ask", response_model=ApiResponse[RagAskResponse])
def ask(request: Request, payload: RagAskRequest) -> ApiResponse[RagAskResponse]:
    logger.info("RAG ask request received (q_chars=%d)", len(payload.question))
    answer, retrieved, note = rag_ask(payload.question, corpus_id=payload.corpus_id)
    logger.info("RAG ask completed (retrieved=%d)", len(retrieved))

    data = RagAskResponse(answer=answer, retrieved=retrieved, note=note)
    return ApiResponse(data=data, meta=_meta(request, mode="rag_ask"))


@router.post("/clear", response_model=ApiResponse[Dict[str, Any]])
def clear(request: Request) -> ApiResponse[Dict[str, Any]]:
    note = rag_clear()
    data = {"status": "ok", "note": note}
    return ApiResponse(data=data, meta=_meta(request, mode="rag_clear"))


@router.delete("/corpora/{corpus_id}", response_model=ApiResponse[Dict[str, Any]])
def delete_corpus(request: Request, corpus_id: str) -> ApiResponse[Dict[str, Any]]:
    note = rag_delete(corpus_id)
    data = {"status": "ok", "note": note, "corpus_id": corpus_id}
    return ApiResponse(data=data, meta=_meta(request, mode="rag_corpus_delete"))
