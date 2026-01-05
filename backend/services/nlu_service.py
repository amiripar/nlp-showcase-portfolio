from __future__ import annotations
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List
from openai import OpenAI

INTENTS = [
    "get_weather",
    "book_flight",
    "book_hotel",
    "order_food",
    "track_order",
    "schedule_meeting",
    "search_information",
    "summarize_text",
    "translate_text",
    "other",
]

ENTITY_TYPES = [
    "PERSON",
    "ORG",
    "LOCATION",
    "DATE",
    "TIME",
    "TIME_OF_DAY",
    "MONEY",
    "NUMBER",
    "PERCENT",
    "PRODUCT",
    "SERVICE",
    "EMAIL",
    "PHONE",
]

def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _find_span(text: str, needle: str) -> Dict[str, int]:
    """
    Try to find offsets for entity text inside the original user text.
    Returns start/end; if not found, returns -1/-1.
    """
    if not needle:
        return {"start": -1, "end": -1}
    i = text.find(needle)
    if i == -1:
        return {"start": -1, "end": -1}
    return {"start": i, "end": i + len(needle)}

def nlu_parse_openai(user_text: str) -> Dict[str, Any]:
    """
    Professional NLU using OpenAI (same style as your NLI task):
    Input: a user utterance
    Output:
      - intent: {name, confidence}
      - entities: [{type, text, start, end, value}]
      - slots: dict
      - needs_clarification: bool
      - clarifying_question: str
      - note: str

    Test in FastAPI docs: http://127.0.0.1:8000/docs  -> POST /nlu/parse
    """
    text = _normalize_text(user_text)

    if not text:
        return {
            "intent": {"name": "other", "confidence": 0.0},
            "entities": [],
            "slots": {},
            "needs_clarification": True,
            "clarifying_question": "Please type your request.",
            "note": "NLU: invalid input (empty text).",
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "intent": {"name": "other", "confidence": 0.0},
            "entities": [],
            "slots": {},
            "needs_clarification": True,
            "clarifying_question": "Backend is missing OPENAI_API_KEY. Please set it in .env.",
            "note": "NLU: missing API key.",
        }

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    now = datetime.now()
    reference_date = now.strftime("%Y-%m-%d")

    system_instructions = (
        "You are an NLU parser for a product.\n"
        "Given a single user text, return:\n"
        "1) intent: choose ONE from the allowed list\n"
        "2) entities: important spans found in the text, with type and normalized value\n"
        "3) slots: a small dictionary mapping arguments for the intent (if possible)\n"
        "4) clarification: if intent is unclear or required slots are missing\n\n"
        "Rules:\n"
        "- Intent must be exactly one of the allowed intent names.\n"
        "- Entity type must be one of the allowed entity types.\n"
        "- Entities must use the exact text span as it appears in the input.\n"
        "- If you can normalize:\n"
        "  * DATE -> ISO-8601 (YYYY-MM-DD) when possible.\n"
        "  * TIME -> HH:MM (24h) when possible.\n"
        "  * MONEY -> keep numeric value + currency if known.\n"
        "- Use the provided reference date to resolve relative dates (today/tomorrow/next Monday, etc).\n"
        "- Confidence values must be between 0 and 1.\n"
        "- If unsure, choose intent 'other' and ask ONE clarifying question.\n"
        "- Keep clarifying_question empty if needs_clarification is false.\n"
        "- Output ONLY valid JSON."
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "enum": INTENTS},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["name", "confidence"],
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "enum": ENTITY_TYPES},
                        "text": {"type": "string"},
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "value": {"type": ["string", "number", "boolean", "null"]},
                    },
                    "required": ["type", "text", "start", "end", "value"],
                },
            },
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": ["string", "number", "boolean", "null"]},
                    },
                    "required": ["key", "value"],
                },
            },

            "needs_clarification": {"type": "boolean"},
            "clarifying_question": {"type": "string"},
        },
        "required": ["intent", "entities", "slots", "needs_clarification", "clarifying_question"],
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": (
                    f"REFERENCE_DATE: {reference_date}\n"
                    f"USER_TEXT:\n{text}\n\n"
                    f"ALLOWED_INTENTS: {', '.join(INTENTS)}\n"
                    f"ALLOWED_ENTITY_TYPES: {', '.join(ENTITY_TYPES)}"
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "nlu_parse_result",
                "schema": schema,
            }
        },
        temperature=0,
    )

    raw_text = getattr(resp, "output_text", None)

    if not raw_text:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                c_type = getattr(c, "type", "")
                if c_type in ("output_text", "text"):
                    parts.append(getattr(c, "text", "") or "")
        raw_text = "".join(parts).strip()

    if not raw_text:
        raise RuntimeError("OpenAI response did not contain output_text to parse.")

    data = json.loads(raw_text)
    
    slots_list = data.get("slots", []) or []
    slots_dict = {item["key"]: item.get("value") for item in slots_list if "key" in item}

    fixed_entities = []
    for ent in data.get("entities", []) or []:
        ent_text = ent.get("text", "")
        span = _find_span(text, ent_text)

        start = ent.get("start", -1)
        end = ent.get("end", -1)

        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < 0:
            start, end = span["start"], span["end"]
        else:
            if start >= 0 and end > start and end <= len(text):
                if text[start:end] != ent_text:
                    start, end = span["start"], span["end"]
            else:
                start, end = span["start"], span["end"]

        fixed_entities.append(
            {
                "type": ent.get("type", "SERVICE"),
                "text": ent_text,
                "start": start,
                "end": end,
                "value": ent.get("value", None),
            }
        )

    needs = bool(data.get("needs_clarification", False))
    cq = (data.get("clarifying_question", "") or "").strip()
    if needs and not cq:
        cq = "Could you clarify what you want to do?"
    if not needs:
        cq = ""

    raw_slots = data.get("slots", {}) or {}

    slots_dict = {}
    if isinstance(raw_slots, list):
        for item in raw_slots:
            if isinstance(item, dict) and "key" in item:
                slots_dict[str(item["key"])] = item.get("value", None)
    elif isinstance(raw_slots, dict):
        slots_dict = raw_slots

    return {
        "intent": data["intent"],
        "entities": fixed_entities,
        "slots": slots_dict,
        "needs_clarification": needs,
        "clarifying_question": cq,
        "note": f"NLU: OpenAI ({model}) structured output.",
    }
