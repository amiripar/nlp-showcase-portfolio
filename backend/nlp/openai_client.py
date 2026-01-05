from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from backend.core.config import settings

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 64
_MAX_RETRIES = 3

_BASE_SLEEP_SEC = 1.0        
_MAX_SLEEP_SEC = 8.0           

_MAX_EMBED_CHARS = 6000

_OPENAI_TIMEOUT_SEC = float(getattr(settings, "openai_timeout_sec", 45.0))

_HTTPX_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=_OPENAI_TIMEOUT_SEC,
    write=10.0,
    pool=10.0,
)

_http_client = httpx.Client(timeout=_HTTPX_TIMEOUT)

_client = OpenAI(
    api_key=settings.openai_api_key,
    timeout=_OPENAI_TIMEOUT_SEC,
    max_retries=0,
    http_client=_http_client,
)

def get_client() -> OpenAI:
    return _client

def _clean_text_list(texts: Sequence[object], max_chars: int = _MAX_EMBED_CHARS) -> List[str]:
    clean: List[str] = []
    for t in (texts or []):
        if t is None:
            continue
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if not t:
            continue
        if len(t) > max_chars:
            t = t[:max_chars]
        clean.append(t)
    return clean

def _sleep_backoff(attempt: int) -> None:
    """
    Jittered exponential backoff:
      sleep = min(MAX, BASE * 2^(attempt-1)) * (0.75..1.25)
    """
    expo = _BASE_SLEEP_SEC * (2 ** max(0, attempt - 1))
    sleep_sec = min(_MAX_SLEEP_SEC, expo)
    jitter = random.uniform(0.75, 1.25)
    time.sleep(sleep_sec * jitter)

def _call_with_retries(fn, *, label: str) -> Any:
    """
    Wrap OpenAI calls with:
    - retries for connection/timeout/rate limit
    - retries for retryable 5xx
    - jittered backoff
    - latency logging
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            out = fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("%s succeeded (attempt %d/%d, %.0f ms)", label, attempt, _MAX_RETRIES, dt_ms)
            return out

        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            last_err = e
            logger.warning(
                "%s failed (attempt %d/%d, %.0f ms): %s",
                label, attempt, _MAX_RETRIES, dt_ms, type(e).__name__,
            )
            if attempt == _MAX_RETRIES:
                raise
            _sleep_backoff(attempt)

        except APIStatusError as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            last_err = e
            status = getattr(e, "status_code", None)
           
            retryable = status is not None and 500 <= status <= 599

            logger.warning(
                "%s status error (attempt %d/%d, %.0f ms, status=%s, retryable=%s)",
                label, attempt, _MAX_RETRIES, dt_ms, status, retryable,
            )
            if not retryable or attempt == _MAX_RETRIES:
                raise
            _sleep_backoff(attempt)

    if last_err:
        raise last_err
    raise RuntimeError(f"{label} failed for unknown reasons.")

def embed_texts(texts: List[str]) -> List[List[float]]:
    clean = _clean_text_list(texts)
    if not clean:
        raise ValueError("embed_texts() received no valid inputs after cleaning.")

    out: List[List[float]] = []
    total = len(clean)

    model = getattr(settings, "openai_embedding_model", None)
    if not model:
        raise ValueError("settings.openai_embedding_model is missing (check .env).")

    logger.info(
        "Embedding %d texts (batch_size=%d, model=%s, timeout=%.0fs)",
        total, _EMBED_BATCH_SIZE, model, _OPENAI_TIMEOUT_SEC,
    )

    for start in range(0, total, _EMBED_BATCH_SIZE):
        batch = clean[start: start + _EMBED_BATCH_SIZE]

        def _do():
            return _client.embeddings.create(model=model, input=batch)

        resp = _call_with_retries(_do, label="OpenAI embeddings.create")
        out.extend([d.embedding for d in resp.data])

    if len(out) != len(clean):
        raise RuntimeError(f"Embeddings count mismatch: got {len(out)} for {len(clean)} inputs.")

    return out

def _llm_text(system: str, user: str, *, model: Optional[str] = None) -> str:
    use_model = model or getattr(settings, "openai_chat_model", "gpt-4.1-mini")

    def _do():
        return _client.responses.create(
            model=use_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    resp = _call_with_retries(_do, label=f"OpenAI responses.create (model={use_model})")
    text = getattr(resp, "output_text", "") or ""
    return text.strip()


def rag_answer_with_llm(question: str, contexts: List[str]) -> str:
    q = (question or "").strip()
    ctx = _clean_text_list(contexts, max_chars=8000)

    if not q:
        return "Invalid question (empty)."
    if not ctx:
        return "Not enough information in the context."

    joined = "\n\n---\n\n".join(ctx[:12])

    system = (
        "You are a retrieval-augmented assistant. Answer ONLY using the provided context.\n"
        "If the context does not contain the answer, respond exactly with:\n"
        "Not enough information in the context.\n"
        "Do not guess or hallucinate.\n"
        "If the user asks for a table of contents or chapter list, output a clean bullet list."
    )

    user = f"QUESTION:\n{q}\n\nCONTEXT:\n{joined}"
    return _llm_text(system, user)


def classify_with_llm(text: str, labels: List[str], top_k: int = 1) -> Dict[str, Any]:
    """
    Stable return shape:
    {
      "label": str,
      "confidence": float,   # 0.0 - 1.0
      "top_k": [{"label": str, "score": float}],
      "note": str
    }
    """
    t = (text or "").strip()
    if not t:
        return {"label": "", "confidence": 0.0, "top_k": [], "note": "Empty input."}
    if not labels:
        return {"label": "", "confidence": 0.0, "top_k": [], "note": "No labels provided."}

    system = (
        "You are a text classifier.\n"
        "Choose exactly one label from the provided label set.\n"
        "Return VALID JSON only with keys: label, confidence, note.\n"
        "confidence MUST be a NUMBER between 0 and 1 (not a percent, not a string).\n"
        "Example: {\"label\":\"positive\",\"confidence\":0.87,\"note\":\"short reason\"}\n"
    )
    user = f"LABELS:\n{labels}\n\nTEXT:\n{t}\n\nReturn JSON only."

    raw = _llm_text(system, user)

    def _normalize_confidence(value: Any) -> float:
        try:
            if isinstance(value, str):
                s = value.strip()
                if s.endswith("%"):
                    s = s[:-1].strip()
                    conf = float(s.replace(",", ".")) / 100.0
                else:
                    conf = float(s.replace(",", "."))
            else:
                conf = float(value)

            if conf > 1.0 and conf <= 100.0:
                conf = conf / 100.0

            return max(0.0, min(1.0, conf))
        except Exception:
            return 0.0

    try:
        obj = json.loads(raw)
        label = str(obj.get("label", "")).strip()
        note = str(obj.get("note", "")).strip()
        confidence = _normalize_confidence(obj.get("confidence", 0.0))

        if label and label not in labels:
            logger.warning("LLM returned label not in label set: %r (allowed=%r)", label, labels)

        top_k_list: List[Dict[str, Any]] = [{"label": label, "score": confidence}] if label else []

        return {
            "label": label,
            "confidence": confidence,
            "top_k": top_k_list,
            "note": note,
        }
    except Exception:
        return {
            "label": "",
            "confidence": 0.0,
            "top_k": [],
            "note": f"LLM returned non-JSON output. Raw: {str(raw)[:200]}",
        }


def summarize_with_llm(text: str, max_chars: int = 600) -> str:
    """
    Matches summarization_service expectations:
    - accepts max_chars
    - returns STRING summary
    """
    t = (text or "").strip()
    if not t:
        return ""

    system = (
        "You are a helpful assistant that summarizes text.\n"
        "Return plain text only (no JSON, no markdown).\n"
        f"Keep the summary under {max_chars} characters.\n"
    )
    user = f"TEXT:\n{t}\n\nSummary:"

    summary = _llm_text(system, user)
    return (summary or "").strip()


def translate_with_llm(text: str, source_lang: str, target_lang: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    src = (source_lang or "").strip() or "auto"
    tgt = (target_lang or "").strip()
    if not tgt:
        return ""

    system = (
        "You are a translation engine.\n"
        "Translate the given text from source language to target language.\n"
        "Return plain text only (no JSON, no markdown).\n"
    )
    user = (
        f"SOURCE_LANG: {src}\n"
        f"TARGET_LANG: {tgt}\n\n"
        f"TEXT:\n{t}\n"
    )

    out = _llm_text(system, user)
    return (out or "").strip()


def chat_with_llm(message: Any) -> str:
    """
    Compatible with chat_service.py:
      - accepts str OR List[Dict[str, str]]
      - returns STRING reply
    """
    if isinstance(message, list):
        messages: List[Dict[str, str]] = []
        for m in message:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if role and content:
                messages.append({"role": role, "content": content})

        if not messages:
            return ""

        system = "You are a helpful chat assistant. Be concise and clear."
        input_items = [{"role": "system", "content": system}] + messages
        use_model = getattr(settings, "openai_chat_model", "gpt-4.1-mini")

        def _do():
            return _client.responses.create(model=use_model, input=input_items)

        resp = _call_with_retries(_do, label=f"OpenAI responses.create (chat, model={use_model})")
        text = getattr(resp, "output_text", "") or ""
        return text.strip()

    msg = (str(message) if message is not None else "").strip()
    if not msg:
        return ""

    system = "You are a helpful chat assistant. Be concise and clear."
    return _llm_text(system, msg)
