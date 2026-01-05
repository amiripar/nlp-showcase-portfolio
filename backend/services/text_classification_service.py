from typing import Dict, Any, List
from backend.core.config import settings
from backend.nlp.openai_client import classify_with_llm

def classify_text(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "label": "",
            "confidence": 0.0,
            "top_k": [],
            "note": "Invalid input (empty text).",
        }

    labels: List[str] = [
        x.strip()
        for x in (settings.classification_labels or "").split(",")
        if x.strip()
    ]
    top_k = max(1, int(getattr(settings, "classification_top_k", 1)))

    if settings.use_llm_classification:
        result = classify_with_llm(cleaned, labels=labels, top_k=top_k)

        label = result.get("label") or (labels[0] if labels else "unknown")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        top_k_list = result.get("top_k") or []
        note = result.get("note") or "LLM classification"

        return {
            "label": label,
            "confidence": confidence,
            "top_k": top_k_list,
            "note": note,
        }

    dummy_label = "neutral"
    dummy_confidence = 1.0
    return {
        "label": dummy_label,
        "confidence": dummy_confidence,
        "top_k": [{"label": dummy_label, "score": dummy_confidence}],
        "note": "Dummy classification",
    }
