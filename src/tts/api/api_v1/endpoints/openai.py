"""
OpenAI-compatible transcription endpoint.

- Accepts multipart/form-data with `file`, plus optional fields
  (model, language, temperature, response_format, prompt, timestamp_granularities).
- Returns **exactly** the shape OpenAI expects for JSON: {"text": "..."}.
- Everything else (decode, resample, VAD, model inference) happens in AudioEngine.

If you want additional metadata, expose another route (e.g., /v1/audio/transcriptions/verbose)
that returns the full dict from the engine.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from core.logging import logger_asr as logger
from core.settings import get_settings
from tts.engine_factory import get_audio_engine
from tts.schemas.audio_engine import ModelResponse, ModelsResponse
from tts.schemas.openai import OpenAICreateSpeechRequest

router = APIRouter()

settings = get_settings()
engine = get_audio_engine()


@router.get("/audio/voices")
async def list_voices() -> Any:
    try:
        pass

    except Exception as e:
        msg = f"OpenAI Transcription Error: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg) from e


# Reference: https://platform.openai.com/docs/api-reference/models/list
@router.get("/models", response_model=ModelsResponse)
async def list_models() -> Any:
    try:
        return engine.list_models()
    except Exception as e:
        msg = f"OpenAI Transcription Error: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg) from e


# Reference: https://platform.openai.com/docs/api-reference/models/retrieve
@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str) -> Any:
    try:
        return engine.get_model(model_id=model_id)
    except Exception as e:
        msg = f"OpenAI Transcription Error: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg) from e


# Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
@router.post("/audio/speech")
async def tts(request: OpenAICreateSpeechRequest) -> Any:
    try:
        result = engine.speech(
            input=request.input,
            voice=request.voice,
            response_format=request.response_format,
            speed=request.speed,
            stream_format=request.stream_format,
        )

        if request.stream_format == "audio":
            return engine.helper_return_audio_file(audio=result, response_format=request.response_format)
        else:
            return engine.helper_return_sse_stream(audio=result, already_sse=True)

    except Exception as e:
        msg = f"OpenAI Transcription Error: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg) from e
