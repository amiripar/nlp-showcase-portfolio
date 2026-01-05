from typing import Dict, List, Literal, TypedDict

Role = Literal["user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


_sessions: Dict[str, List[ChatMessage]] = {}


def get_history(session_id: str) -> List[ChatMessage]:
    return _sessions.get(session_id, [])


def append_message(session_id: str, role: Role, content: str) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
