from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from asr.engine_factory import get_audio_engine
from asr.schemas.audio_engine import DetectedLanguage, ListModelsResponse, Output
from core.logging import logger_asr as logger
from core.settings import get_settings

router = APIRouter()
settings = get_settings()
engine = get_audio_engine()


@router.post(
    "/asr",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "application/x-subrip": {},
                "text/vtt; charset=utf-8": {},
                "text/plain; charset=utf-8": {},
                "text/tab-separated-values; charset=utf-8": {},
                "application/jsonl; charset=utf-8": {},
            },
            "description": "Bazarr-compatible ASR outputs (JSON subtitles payload or subtitle text).",
        }
    },
)
async def bazarr_asr(
    file: UploadFile = File(...),
    task: str = Query(default="transcribe", description="Operation: transcribe or translate"),
    language: str | None = Query(default=None, description="Language of the audio if known"),
    initial_prompt: str | None = Query(default=None, description="Prompt text; ignored for Whisper"),
    encode: bool = Query(default=True, description="IGNORED: Encode audio first through ffmpeg"),
    output: Output | None = Query(default=Output.TXT, description="Output formats: txt|vtt|srt|tsv|json|jsonl"),
    video_file: str | None = Query(default=None, description="Original video file path for logging purposes"),
) -> Any:
    """
    Returns segments suitable for creating subtitles in Bazarr workflows.
     JSON shape (when output=json):
      { "language": "en", "segments": [ { "start": 0.0, "end": 1.2, "text": "..." }, ... ] }
    """
    try:
        # Read entire uploaded audio bytes into memory
        audio = await file.read()
        size_b = len(audio)

        # default to TXT if output is None (defensive)
        output_format: Output = output or Output.TXT

        logger.info(
            "ASR/Bazarr request: filename=%s video_file=%s size=%dB lang_hint=%s output=%s task=%s",
            file.filename,
            video_file,
            size_b,
            language,
            output_format,
            task,
        )

        result = engine.transcribe_file(
            file_bytes=audio,
            filename=file.filename,
            request_language=language,
            beam_size=5,
            temperature=0.0,
            best_of=1,
        )

        return engine.helper_write_output(file=file, result=result, output=output_format)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bazarr ASR error: {e}") from e


@router.post(
    "/detect-language",
    response_model=DetectedLanguage,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": DetectedLanguage.model_json_schema()},
            },
            "description": "JSON (Bazarr-compatible) for detected language response.",
        }
    },
)
async def detect_language(
    file: UploadFile = File(...),
    encode: bool = Query(default=True, description="IGNORED: Encode audio first through ffmpeg"),
    detect_lang_length: int | None = Query(default=None, description="Detect language on X seconds of the file"),
    detect_lang_offset: int | None = Query(default=None, description="Start Detect language X seconds into the file"),
    video_file: str | None = Query(default=None, description="Original video file path for logging purposes"),
) -> Any:
    """
    Detect the Language of Audio for Bazarr
    """
    try:
        audio = await file.read()

        logger.info(
            "Detect Language request: filename=%s video_file=%s detect_lang_length=%s detect_lang_offset=%s",
            file.filename,
            video_file,
            detect_lang_length,
            detect_lang_offset,
        )

        result = engine.detect_language_file(
            file_bytes=audio,
            filename=file.filename,
            detect_lang_length=detect_lang_length,
            detect_lang_offset=detect_lang_offset,
        )

        return DetectedLanguage(**result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR error: {e}") from e


@router.get("/models", response_model=ListModelsResponse)
def list_models() -> Any:
    """
    OpenAI-compatible models listing. Delegates to the engine so other
    API mocks (or future services) can reuse the same definition.
    """
    return ListModelsResponse(**engine.list_models())
