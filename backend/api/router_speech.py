import logging
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field

from backend.services.speech_service import transcribe_audio, synthesize_speech_base64

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/speech",
    tags=["Speech (ASR/TTS)"],
)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize (cannot be empty).")


class TTSResponse(BaseModel):
    audio_base64: str
    mime_type: str
    note: str


class ASRResponse(BaseModel):
    transcript: str
    note: str


@router.post("/asr", response_model=ASRResponse)
async def asr(audio: UploadFile = File(...)):
    logger.info("ASR request received (filename=%s, content_type=%s)", audio.filename, audio.content_type)

    audio_bytes = await audio.read()
    transcript, note = transcribe_audio(audio_bytes, audio.filename or "audio.wav")

    logger.info("ASR completed (transcript_chars=%d)", len(transcript))
    return ASRResponse(transcript=transcript, note=note)



@router.post("/tts", response_model=TTSResponse)
def tts(payload: TTSRequest):
    logger.info("TTS request received (chars=%d)", len(payload.text))

    audio_base64, mime_type, note = synthesize_speech_base64(payload.text)

    logger.info("TTS completed (audio_bytes_base64_chars=%d)", len(audio_base64))
    return TTSResponse(audio_base64=audio_base64, mime_type=mime_type, note=note)
