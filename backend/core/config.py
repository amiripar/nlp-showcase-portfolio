import os
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "NLP Showcase API"
    app_version: str = "0.1.0"

    environment: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    log_file: str = "logs/app.log"

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.2")
    openai_embedding_model: str = "text-embedding-3-small"

    use_llm_summarization: bool = os.getenv("USE_LLM_SUMMARIZATION", "false").lower() == "true"

    chat_max_turns: int = int(os.getenv("CHAT_MAX_TURNS", "8"))
    use_llm_chat: bool = os.getenv("USE_LLM_CHAT", "false").lower() == "true"

    use_llm_qa: bool = os.getenv("USE_LLM_QA", "false").lower() == "true"
    qa_max_context_chars: int = int(os.getenv("QA_MAX_CONTEXT_CHARS", "4000"))
    qa_model: str = os.getenv("QA_MODEL", "gpt-4o-mini")

    use_semantic_ir: bool = os.getenv("USE_SEMANTIC_IR", "false").lower() == "true"
    ir_embedding_model: str = os.getenv("IR_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    use_llm_ner: bool = os.getenv("USE_LLM_NER", "false").lower() == "true"
    ner_model: str = os.getenv("NER_MODEL", "gpt-4o-mini")
    ner_max_text_chars: int = int(os.getenv("NER_MAX_TEXT_CHARS", "8000"))

    use_rag: bool = os.getenv("USE_RAG", "false").lower() == "true"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    use_llm_classification: bool = os.getenv("USE_LLM_CLASSIFICATION", "false").lower() == "true"
    classification_labels: str = os.getenv("CLASSIFICATION_LABELS", "positive,neutral,negative")
    classification_top_k: int = int(os.getenv("CLASSIFICATION_TOP_K", "3"))

    use_llm_translation: bool = os.getenv("USE_LLM_TRANSLATION", "false").lower() == "true"

    speech_asr_model: str = os.getenv("SPEECH_ASR_MODEL", "gpt-4o-mini-transcribe")
    speech_tts_model: str = os.getenv("SPEECH_TTS_MODEL", "gpt-4o-mini-tts")
    speech_tts_voice: str = os.getenv("SPEECH_TTS_VOICE", "alloy")
    speech_tts_format: str = os.getenv("SPEECH_TTS_FORMAT", "mp3")


settings = Settings()
