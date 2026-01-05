from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR

from backend.api.schemas import ApiErrorResponse, ErrorPayload, Meta


def _meta(request: Request) -> Meta:
    rid = getattr(request.state, "request_id", "unknown")
    return Meta(request_id=rid)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = ApiErrorResponse(
        error=ErrorPayload(
            code="validation_error",
            message="Invalid request body",
            details={"errors": exc.errors()},
        ),
        meta=_meta(request),
    )
    return JSONResponse(status_code=HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    payload = ApiErrorResponse(
        error=ErrorPayload(
            code="internal_error",
            message="Unexpected server error",
        ),
        meta=_meta(request),
    )
    return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())
