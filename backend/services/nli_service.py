from __future__ import annotations
import json
import os
import re
from typing import Any, Dict
from openai import OpenAI

def _normalize_text(s: str) -> str:
    """Small normalization to keep prompts stable."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def nli_predict_openai(premise: str, hypothesis: str) -> Dict[str, Any]:
    """
    Professional NLI using OpenAI (same style as your other tasks):
    Input: premise + hypothesis
    Output: label (entailment/contradiction/neutral) + confidence (0..1) + short rationale

    Tested via FastAPI docs: http://127.0.0.1:8000/docs
    """

    premise = _normalize_text(premise)
    hypothesis = _normalize_text(hypothesis)

    if not premise:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "rationale": "Premise is empty.",
            "note": "NLI: invalid input (empty premise).",
        }

    if not hypothesis:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "rationale": "Hypothesis is empty.",
            "note": "NLI: invalid input (empty hypothesis).",
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "rationale": "OPENAI_API_KEY is not set in backend environment.",
            "note": "NLI: missing API key.",
        }

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    system_instructions = (
        "You are an NLI classifier.\n"
        "Decide the relationship between a PREMISE and a HYPOTHESIS.\n"
        "Return one label:\n"
        "- entailment: hypothesis is guaranteed true from the premise\n"
        "- contradiction: hypothesis is false / impossible given the premise\n"
        "- neutral: neither entailed nor contradicted\n\n"
        "Be strict: if the premise does not fully guarantee the hypothesis, choose neutral.\n"
        "Give a short rationale (max 1 sentence).\n"
        "Output ONLY valid JSON."
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["label", "confidence", "rationale"],
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": f"PREMISE:\n{premise}\n\nHYPOTHESIS:\n{hypothesis}",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "nli_result",
                "schema": schema,
            }
        },
        temperature=0,
    )

    raw_text = getattr(resp, "output_text", None)

    if not raw_text:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                c_type = getattr(c, "type", "")
                if c_type in ("output_text", "text"):
                    parts.append(getattr(c, "text", "") or "")
        raw_text = "".join(parts).strip()

    if not raw_text:
        raise RuntimeError("OpenAI response did not contain output_text to parse.")

    data = json.loads(raw_text)

    return {
        "label": data["label"],
        "confidence": float(data["confidence"]),
        "rationale": data["rationale"],
        "note": f"NLI: OpenAI ({model}) structured output.",
    }
