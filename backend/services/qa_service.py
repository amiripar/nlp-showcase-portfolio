from __future__ import annotations
import re
from typing import List, Tuple
from backend.core.config import settings

UNKNOWN_ANSWER = "I don't know based on the provided context."

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]

def answer_question_grounded_dummy(context: str, question: str) -> Tuple[str, str, str]:
    context = (context or "").strip()
    question = (question or "").strip()

    if not context:
        return UNKNOWN_ANSWER, "", "Dummy QA: empty context."
    if not question:
        return UNKNOWN_ANSWER, "", "Dummy QA: empty question."

    keywords = re.findall(r"[A-Za-z0-9\u0600-\u06FF']+", question.lower())
    keywords = [k for k in keywords if len(k) >= 2]

    sentences = _split_sentences(context)
    if not sentences:
        return UNKNOWN_ANSWER, "", "Dummy QA: insufficient context."

    best_sentence = ""
    best_score = 0

    for s in sentences:
        s_l = s.lower()
        score = sum(1 for k in keywords if k in s_l)
        if score > best_score:
            best_score = score
            best_sentence = s

    if best_score == 0:
        return UNKNOWN_ANSWER, "", "Dummy QA: no supporting evidence found in context."

    return best_sentence, best_sentence, "Dummy QA: answered using keyword match from context."

def _truncate_context(context: str) -> str:
    context = (context or "").strip()
    max_chars = int(getattr(settings, "qa_max_context_chars", 4000) or 4000)
    if len(context) > max_chars:
        return context[:max_chars]
    return context

def _normalize_llm_answer(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return UNKNOWN_ANSWER

    low = a.lower()
    refusal_signals = [
        "not enough information",
        "insufficient information",
        "not enough info",
        "cannot be determined",
        "can't be determined",
        "not provided in the context",
        "not in the context",
        "not mentioned in the context",
        "i don't know",
        "i do not know",
        "i cannot answer",
        "unable to determine",
        "not specified",
    ]
    if any(sig in low for sig in refusal_signals):
        return UNKNOWN_ANSWER

    return a

def _answer_question_grounded_openai(context: str, question: str) -> Tuple[str, str, str]:
    try:
        from openai import OpenAI
    except Exception as e:
        ans, ev, note = answer_question_grounded_dummy(context, question)
        return ans, ev, f"{note} (OpenAI SDK not available: {e})"

    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        ans, ev, note = answer_question_grounded_dummy(context, question)
        return ans, ev, f"{note} (OPENAI_API_KEY missing)"

    model = (getattr(settings, "qa_model", "") or "gpt-4o-mini").strip()
    context = _truncate_context(context)

    system_prompt = (
        "You are a grounded QA assistant.\n"
        "Rules:\n"
        "1) Answer ONLY using the CONTEXT.\n"
        f"2) If the answer is not in the CONTEXT, reply exactly: {UNKNOWN_ANSWER}\n"
        "3) Also return EVIDENCE: copy ONE sentence from the CONTEXT that supports the answer.\n"
        "4) Output format MUST be exactly:\n"
        "ANSWER: <one short answer>\n"
        "EVIDENCE: <one sentence copied from context>\n"
        "5) Do not add any other text, explanation, or formatting."
    )

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    client = OpenAI(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        raw = completion.choices[0].message.content or ""
    except Exception as e:
        ans, ev, note = answer_question_grounded_dummy(context, question)
        return ans, ev, f"{note} (OpenAI API error: {e})"

    final_answer = UNKNOWN_ANSWER
    final_evidence = ""

    answer_match = re.search(r"(?im)^ANSWER:\s*(.*?)(?:\n|$)", raw)
    evidence_match = re.search(r"(?im)^EVIDENCE:\s*(.*?)(?:\n|$)", raw)

    if answer_match:
        final_answer = _normalize_llm_answer(answer_match.group(1).strip())
    else:
        final_answer = _normalize_llm_answer(raw)

    if final_answer != UNKNOWN_ANSWER:
        if evidence_match:
            final_evidence = evidence_match.group(1).strip()
        else:
            final_evidence = ""

    return (
        final_answer,
        final_evidence,
        f"LLM QA: answered using OpenAI model ({model}), grounded to provided context.",
    )

def answer_question_grounded(context: str, question: str) -> Tuple[str, str, str]:
    context = (context or "").strip()
    question = (question or "").strip()

    if not context or not question:
        return UNKNOWN_ANSWER, "", "QA: missing context or question."

    if bool(getattr(settings, "use_llm_qa", False)):
        return _answer_question_grounded_openai(context, question)

    return answer_question_grounded_dummy(context, question)