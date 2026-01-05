from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, Request
from pydantic import BaseModel, Field
from typing import List

from backend.services.ir_service import build_index, search
from backend.api.schemas import ApiResponse, Meta
from backend.core.config import settings

router = APIRouter(prefix="/ir", tags=["IR"])

class IndexResponse(BaseModel):
    num_files: int
    num_chunks: int
    vocab_size: int
    semantic_ready: bool
    note: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    mode: str = Field("auto", description="keyword | semantic | auto")


class SearchResult(BaseModel):
    doc_name: str
    chunk_id: int
    page_num: int | None
    score: float
    text: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    note: str


@router.post("/index", response_model=ApiResponse[IndexResponse])
async def ir_index(
    request: Request,
    files: List[UploadFile] = File(...),
    chunk_size: int = 900,
    overlap: int = 150,
) -> ApiResponse[IndexResponse]:
    loaded = []
    for f in files:
        data = await f.read()
        loaded.append((f.filename, data))

    info = build_index(loaded, chunk_size=chunk_size, overlap=overlap)

    data = IndexResponse(**info, note="IR: index built successfully.")
    meta = Meta(
        request_id=request.state.request_id,
        version=getattr(settings, "API_VERSION", "0.1.0"),
        mode="index",
    )
    return ApiResponse(data=data, meta=meta)


@router.post("/search", response_model=ApiResponse[SearchResponse])
def ir_search(request: Request, req: SearchRequest) -> ApiResponse[SearchResponse]:
    out = search(req.query, top_k=req.top_k, mode=req.mode)

    data = SearchResponse(results=out["results"], note=out["note"])

    meta = Meta(
        request_id=request.state.request_id,
        version=getattr(settings, "API_VERSION", "0.1.0"),
        mode=req.mode,            
        model=out.get("model"),   
    )

    return ApiResponse(data=data, meta=meta)











