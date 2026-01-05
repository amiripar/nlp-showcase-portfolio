import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.chat_service import chat_reply_dummy
from backend.core.session import clear_session, get_history

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Dialogue / Conversational AI"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=64, description="Session identifier.")
    message: str = Field(..., min_length=1, description="User message.")


class ChatResponse(BaseModel):
    reply: str
    session_size: int
    note: str


@router.post("/respond", response_model=ChatResponse)
def respond(payload: ChatRequest) -> ChatResponse:
    logger.info("Chat request received (session_id=%s, msg_chars=%d)", payload.session_id, len(payload.message))

    reply, session_size, note = chat_reply_dummy(payload.session_id, payload.message)

    logger.info("Chat completed (session_id=%s, session_size=%d)", payload.session_id, session_size)

    return ChatResponse(reply=reply, session_size=session_size, note=note)


@router.post("/clear")
def clear(payload: dict):
    """
    Clears chat history for a given session_id.
    Body: { "session_id": "abc123" }
    """
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return {"status": "error", "message": "session_id is required"}

    clear_session(session_id)
    logger.info("Chat session cleared (session_id=%s)", session_id)
    return {"status": "ok", "message": f"Session '{session_id}' cleared."}


@router.get("/history/{session_id}")
def history(session_id: str):
    """
    Returns chat history for debugging/demo.
    """
    hist = get_history(session_id)
    return {"session_id": session_id, "history": hist, "count": len(hist)}
