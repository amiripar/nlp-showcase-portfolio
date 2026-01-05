from backend.core.config import settings
from backend.nlp.openai_client import translate_with_llm

def translate_text_tagged(text: str, source_lang: str, target_lang: str):
    cleaned = (text or "").strip()
    if not cleaned:
        return "", "Invalid input (empty text)."

    if settings.use_llm_translation:
        translated = translate_with_llm(cleaned, source_lang, target_lang)
        return translated, "LLM translation"

    return cleaned, "Dummy translation (echo)"

