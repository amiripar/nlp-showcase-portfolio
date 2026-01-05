from typing import Tuple
import base64
from backend.core.config import settings
from backend.nlp.openai_client import get_client

def transcribe_audio(audio_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Real ASR using OpenAI Audio Transcriptions.
    Returns: (transcript, note)
    """
    if not audio_bytes:
        return "", "ASR: empty audio payload."

    client = get_client()

    result = client.audio.transcriptions.create(
        model=settings.speech_asr_model,
        file=(filename or "audio.wav", audio_bytes),
        response_format="json",
    )

    transcript = (getattr(result, "text", "") or "").strip()
    note = f"ASR: transcribed with OpenAI model={settings.speech_asr_model}."
    return transcript, note

def synthesize_speech_base64(text: str) -> Tuple[str, str, str]:
    """
    Real TTS using OpenAI Audio Speech generation.
    Returns: (audio_base64, mime_type, note)
    """
    text = (text or "").strip()
    if not text:
        return "", "audio/mpeg", "TTS: empty text."

    client = get_client()

    response = client.audio.speech.create(
        model=settings.speech_tts_model,
        voice=settings.speech_tts_voice,
        input=text,
        response_format=settings.speech_tts_format,
    )

    audio_bytes = response.read()

    fmt = (settings.speech_tts_format or "mp3").lower()
    mime_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    note = f"TTS: generated with OpenAI model={settings.speech_tts_model}, voice={settings.speech_tts_voice}, format={fmt}."
    return audio_b64, mime_type, note
