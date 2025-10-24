from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from asr.engine_factory import get_audio_engine
from asr.schemas.audio_engine import DetectedLanguage, ListModelsResponse, Output
from asr.schemas.bazarr import BazarrAsrRequest, BazarrDetectLanguageRequest
from core.logging import logger_asr as logger
from core.settings import get_settings

router = APIRouter()
settings = get_settings()
engine = get_audio_engine()


@router.post(
    "/asr",
    response_model=BazarrAsrRequest,
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
    request: BazarrAsrRequest = Depends(BazarrAsrRequest.as_form),
    file: UploadFile = File(...),
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

        # default to TXT if request.output is None (defensive)
        output: Output = request.output or Output.TXT

        logger.info(
            "ASR/Bazarr request: filename=%s size=%dB lang_hint=%s output=%s",
            getattr(file, "filename", None),
            size_b,
            request.language,
            request.output,
        )

        result = engine.transcribe_file(
            file_bytes=audio,
            filename=file.filename,
            request_language=request.language,
            beam_size=5,
            temperature=0.0,
            best_of=1,
        )

        return engine.helper_write_output(file=file, result=result, output=output)

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
    request: BazarrDetectLanguageRequest = Depends(BazarrDetectLanguageRequest.as_form),
    file: UploadFile = File(...),
) -> Any:
    """
    Detect the Language of Audio for Bazarr
    """
    try:
        audio = await file.read()

        result = engine.detect_language_file(
            file_bytes=audio,
            filename=file.filename,
            detect_lang_length=request.detect_lang_length,
            detect_lang_offset=request.detect_lang_offset,
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
