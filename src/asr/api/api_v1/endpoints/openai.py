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

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from asr.engine import audio_engine
from asr.schemas.audio_engine import ListModelsResponse
from asr.schemas.openai import (
    OpenAITranscription,
    OpenAITranscriptionRequest,
    OpenAITranscriptionVerboseRequest,
    OpenAITranslationsRequest,
    TranscribeVerboseResponse,
)
from core.logging import logger_asr as logger
from core.settings import get_settings

router = APIRouter()

settings = get_settings()


@router.post(
    "/audio/transcriptions",
    response_model=OpenAITranscription,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": OpenAITranscription.model_json_schema()},
                "text/plain": {"schema": {"type": "string"}},
            },
            "description": "JSON (OpenAI-compatible) or plain text transcript",
        }
    },
)
async def transcriptions(
    request: OpenAITranscriptionRequest = Depends(OpenAITranscriptionRequest.as_form),
    file: UploadFile = File(...),
) -> Any:
    try:
        # Read entire uploaded audio bytes into memory
        audio = await file.read()

        result = audio_engine.transcribe_file(
            file_bytes=audio, filename=file.filename, request_language=request.language, temperature=request.temperature
        )

        # If return  is text lets give it a plan text
        if request.response_format == "text":
            return PlainTextResponse(content=result.get("text", ""), status_code=status.HTTP_200_OK)

        # or we default to JSON
        return JSONResponse({"text": result.get("text", "")})

    except Exception as e:
        msg = f"OpenAI Transcription Error: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg) from e


@router.post(
    "/audio/transcriptions/verbose",
    response_model=TranscribeVerboseResponse,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": TranscribeVerboseResponse.model_json_schema()},
                "text/plain": {"schema": {"type": "string"}},
            },
            "description": "JSON (OpenAI-compatible) or plain text transcript",
        }
    },
)
async def transcriptions_verbose(
    request: OpenAITranscriptionVerboseRequest = Depends(OpenAITranscriptionVerboseRequest.as_form),
    file: UploadFile = File(...),
) -> Any:
    """
    Verbose response: returns the full dict from the engine, including:
      text, segments, language, language_probability, durations, timings, vad_used, model.
    """
    try:
        audio = await file.read()

        result = audio_engine.transcribe_file(
            file_bytes=audio,
            filename=file.filename,
            request_language=request.language,
            task=request.task,
            beam_size=request.beam_size,
            temperature=request.temperature,
            best_of=request.best_of,
            word_timestamps=request.word_timestamps,
            vad=request.vad,
        )

        if request.response_format == "text":
            # If someone still asks for text, mirror the minimal behavior.
            return PlainTextResponse(content=result.get("text", ""), status_code=status.HTTP_200_OK)

        # Full JSON payload for debugging/analytics.
        return JSONResponse(result)

    except Exception as e:
        msg = f"OpenAI Transcription (Verbose) Error: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg) from e


@router.post(
    "/audio/translations",
    response_model=OpenAITranscription,
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {"schema": OpenAITranscription.model_json_schema()},
                "text/plain": {"schema": {"type": "string"}},
            },
            "description": "JSON (OpenAI-compatible) or plain text transcript",
        }
    },
)
async def translations(
    request: OpenAITranslationsRequest = Depends(OpenAITranslationsRequest.as_form),
    file: UploadFile = File(...),
) -> Any:
    """
    OpenAI-compatible Translation endpoint.

    Behavior:
      - Runs Whisper/Faster-Whisper with task="translate" (source → English).
      - Returns OpenAI minimal shape for JSON: {"text": "..."}.
      - If response_format="text", returns plain text instead of JSON.
    """
    try:
        audio = await file.read()

        result = audio_engine.transcribe_file(
            file_bytes=audio,
            filename=file.filename,
            # Let the engine auto-detect source language; translation target is English.
            task="translate",
            temperature=request.temperature,
            # Intentionally don't pass `request_language` here; Whisper detects source.
        )

        if request.response_format == "text":
            return PlainTextResponse(content=result.get("text", ""), status_code=status.HTTP_200_OK)

        return JSONResponse({"text": result.get("text", "")})

    except Exception as e:
        msg = f"OpenAI Translation Error: {e}"
        logger.exception(msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg) from e


@router.get("/models", response_model=ListModelsResponse)
def list_models() -> Any:
    """
    OpenAI-compatible models listing. Delegates to the engine so other
    API mocks (or future services) can reuse the same definition.
    """
    return audio_engine.list_models()
