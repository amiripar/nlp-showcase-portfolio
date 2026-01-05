from __future__ import annotations

from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class Meta(BaseModel):
    request_id: str = Field(..., description="Correlation id for logs and tracing")
    version: str = Field("0.1.0", description="API version")
    mode: Optional[str] = Field(None, description="dummy|llm|hybrid etc.")
    model: Optional[str] = Field(None, description="Model identifier if applicable")


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ApiResponse(BaseModel, Generic[T]):
    ok: bool = True
    data: T
    meta: Meta


class ApiErrorResponse(BaseModel):
    ok: bool = False
    error: ErrorPayload
    meta: Meta
