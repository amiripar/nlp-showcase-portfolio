from typing import Tuple
from backend.core.config import settings
from backend.nlp.openai_client import summarize_with_llm

def summarize_text_simple(text: str, max_chars: int = 120) -> Tuple[str, float]:
    cleaned = text.strip()
    if not cleaned:
        return "", 0.0

    if settings.use_llm_summarization:
        summary = summarize_with_llm(cleaned, max_chars=max_chars)[:max_chars].strip()
    else:
        summary = cleaned[:max_chars].strip()

    compression_ratio = len(summary) / max(len(cleaned), 1)
    return summary, compression_ratio
