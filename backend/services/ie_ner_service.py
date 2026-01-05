from __future__ import annotations
import json
import re
from typing import Dict, List, Tuple
from backend.core.config import settings

SUPPORTED_ENTITY_TYPES = ["PERSON", "ORG", "LOCATION", "DATE", "TIME", "MONEY"]

def _normalize_label(label: str) -> str:
    l = (label or "").strip().upper()

    mapping = {
        "ORGANIZATION": "ORG",
        "COMPANY": "ORG",
        "CORP": "ORG",
        "INSTITUTION": "ORG",

        "LOC": "LOCATION",
        "GPE": "LOCATION",
        "PLACE": "LOCATION",

        "CURRENCY": "MONEY",
        "AMOUNT": "MONEY",
    }

    return mapping.get(l, l)

def _find_spans(pattern: str, text: str, label: str) -> List[Dict]:
    results: List[Dict] = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        results.append(
            {"text": m.group(0), "label": label, "start": m.start(), "end": m.end()}
        )
    return results

def _dedupe_entities(entities: List[Dict]) -> List[Dict]:
    seen = set()
    deduped: List[Dict] = []
    for e in sorted(entities, key=lambda x: (x["start"], x["end"], x["label"])):
        key = (e["label"], e["start"], e["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped

def _truncate_text(text: str) -> str:
    text = (text or "").strip()
    max_chars = int(getattr(settings, "ner_max_text_chars", 8000) or 8000)
    return text if len(text) <= max_chars else text[:max_chars]

def extract_entities_dummy(text: str, entity_types: List[str]) -> Tuple[List[Dict], str]:
    text = (text or "").strip()
    if not text:
        return [], "Dummy NER: empty text."

    selected = [t.upper().strip() for t in (entity_types or []) if t.strip()]
    if not selected:
        selected = SUPPORTED_ENTITY_TYPES[:]

    selected = [t for t in selected if t in SUPPORTED_ENTITY_TYPES]
    if not selected:
        return [], "Dummy NER: no supported entity types selected."

    entities: List[Dict] = []

    if "MONEY" in selected:
        entities += _find_spans(r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?", text, "MONEY")

    if "DATE" in selected:
        entities += _find_spans(r"\b\d{4}-\d{2}-\d{2}\b", text, "DATE")
        entities += _find_spans(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:,\s*\d{4})?\b",
            text,
            "DATE",
        )

    if "TIME" in selected:
        entities += _find_spans(r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b", text, "TIME")

    if "ORG" in selected:
        entities += _find_spans(
            r"\b[A-Z][A-Za-z&.\- ]+\s(?:Inc|Ltd|LLC|Corp|Corporation|University|Bank)\b",
            text,
            "ORG",
        )

    if "LOCATION" in selected:
        entities += _find_spans(
            r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s?[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b",
            text,
            "LOCATION",
        )

    if "PERSON" in selected:
        entities += _find_spans(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", text, "PERSON")

    entities = _dedupe_entities(entities)
    return entities, "Dummy NER: extracted entities using simple regex rules."

def _match_spans_left_to_right(text: str, items: List[Dict]) -> List[Dict]:
    """
    Robust span matching for LLM entities.
    - Matches each entity text anywhere in the original text (case-insensitive fallback)
    - Avoids reusing the same exact span
    - Tries a cleaned version if exact match fails
    """
    used_spans = set()
    results: List[Dict] = []

    def find_span(needle: str):
        if not needle:
            return None

        idx = text.find(needle)
        if idx != -1:
            return idx, idx + len(needle)

        idx = text.lower().find(needle.lower())
        if idx != -1:
            return idx, idx + len(needle)

        return None

    for it in items:
        ent_text = (it.get("text") or "").strip()
        label = _normalize_label(it.get("label"))

        if not ent_text:
            continue
        if label not in SUPPORTED_ENTITY_TYPES:
            continue

        span = find_span(ent_text)

        if span is None:
            cleaned = ent_text.strip(" \t\n\r\"'`.,;:()[]{}")
            span = find_span(cleaned)
            if span is not None:
                ent_text = cleaned

        if span is None:
            continue

        start, end = span
        key = (label, start, end)
        if key in used_spans:
            continue

        used_spans.add(key)
        results.append({"text": text[start:end], "label": label, "start": start, "end": end})

    return _dedupe_entities(results)

def _extract_entities_openai(text: str, entity_types: List[str]) -> Tuple[List[Dict], str]:
    try:
        from openai import OpenAI
    except Exception as e:
        entities, note = extract_entities_dummy(text, entity_types)
        return entities, f"{note} (OpenAI SDK not available: {e})"

    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        entities, note = extract_entities_dummy(text, entity_types)
        return entities, f"{note} (OPENAI_API_KEY missing)"

    model = (getattr(settings, "ner_model", "") or "gpt-4o-mini").strip()

    text = _truncate_text(text)

    selected = [t.upper().strip() for t in (entity_types or []) if t.strip()]
    if not selected:
        selected = SUPPORTED_ENTITY_TYPES[:]
    selected = [t for t in selected if t in SUPPORTED_ENTITY_TYPES]
    if not selected:
        return [], "LLM NER: no supported entity types selected."

    system_prompt = (
        "You are an information extraction system.\n"
        "Task: Named Entity Recognition (NER).\n"
        "Return ONLY JSON.\n"
        "Rules:\n"
        "- Extract only the requested entity types.\n"
        "- Do not invent entities.\n"
        "- Each entity must be an exact substring of the input text.\n"
        "- Output JSON format:\n"
        "- Do NOT return combined entities (e.g., 'OpenAI, Anthropic, Google'). Return them separately.\n"
        '{ "entities": [ {"text": "...", "label": "PERSON|ORG|LOCATION|DATE|TIME|MONEY"}, ... ] }\n'
    )

    user_prompt = (
        f"ENTITY_TYPES: {selected}\n\n"
        f"TEXT:\n{text}\n"
    )

    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = (getattr(resp, "output_text", "") or "").strip()

    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return [], f"LLM NER: could not parse JSON output. Raw: {raw[:200]}"
        obj = json.loads(m.group(0))

    items = obj.get("entities", [])
    if not isinstance(items, list):
        return [], "LLM NER: invalid JSON format (entities is not a list)."

    entities = _match_spans_left_to_right(text, items)

    return entities, f"LLM NER: extracted entities using OpenAI model ({model})."

def extract_entities(text: str, entity_types: List[str]) -> Tuple[List[Dict], str]:
    """
    Main entry:
    - USE_LLM_NER=true  -> OpenAI NER
    - else             -> dummy regex NER
    """
    text = (text or "").strip()
    if not text:
        return [], "NER: empty text."

    if bool(getattr(settings, "use_llm_ner", False)):
        return _extract_entities_openai(text, entity_types)

    return extract_entities_dummy(text, entity_types)
