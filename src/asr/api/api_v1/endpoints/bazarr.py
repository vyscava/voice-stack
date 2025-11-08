from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from asr.engine_factory import acquire_engine, get_audio_engine, release_engine
from asr.schemas.audio_engine import DetectedLanguage, ListModelsResponse, Output
from core.logging import logger_asr as logger
from core.settings import get_settings

router = APIRouter()
settings = get_settings()
# Note: get_audio_engine() is used for model listing (read-only),
# but inference endpoints use acquire_engine() for concurrency control
_engine_for_listing = get_audio_engine()

# Simple in-memory stats tracking
_stats = {
    "started_at": datetime.now().isoformat(),
    "endpoints": {
        "/asr": defaultdict(int),
        "/detect-language": defaultdict(int),
    },
    "total_requests": 0,
    "total_success": 0,
    "total_errors": 0,
}


def _track_request(endpoint: str, status_code: int) -> None:
    """Track request statistics."""
    _stats["total_requests"] += 1
    _stats["endpoints"][endpoint][f"status_{status_code}"] += 1

    if 200 <= status_code < 300:
        _stats["total_success"] += 1
        _stats["endpoints"][endpoint]["success"] += 1
    else:
        _stats["total_errors"] += 1
        _stats["endpoints"][endpoint]["errors"] += 1


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
    audio_file: UploadFile = File(...),
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
    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        # Read entire uploaded audio bytes into memory
        audio = await audio_file.read()
        size_b = len(audio)

        # default to TXT if output is None (defensive)
        output_format: Output = output or Output.TXT

        logger.info(
            "ASR/Bazarr request: filename=%s video_file=%s size=%dB lang_hint=%s output=%s task=%s",
            audio_file.filename,
            video_file,
            size_b,
            language,
            output_format,
            task,
        )

        result = engine.transcribe_file(
            file_bytes=audio,
            filename=audio_file.filename,
            request_language=language,
            beam_size=5,
            temperature=0.0,
            best_of=1,
        )

        _track_request("/asr", 200)
        return engine.helper_write_output(file=audio_file, result=result, output=output_format)

    except Exception as e:
        _track_request("/asr", 500)
        raise HTTPException(status_code=500, detail=f"Bazarr ASR error: {e}") from e
    finally:
        # Always release the concurrency slot
        release_engine()


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
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="IGNORED: Encode audio first through ffmpeg"),
    detect_lang_length: int | None = Query(default=None, description="Detect language on X seconds of the file"),
    detect_lang_offset: int | None = Query(default=None, description="Start Detect language X seconds into the file"),
    video_file: str | None = Query(default=None, description="Original video file path for logging purposes"),
) -> Any:
    """
    Detect the Language of Audio for Bazarr
    """
    # Acquire engine with concurrency control
    engine = await acquire_engine()

    try:
        audio = await audio_file.read()

        logger.info(
            "Detect Language request: filename=%s video_file=%s detect_lang_length=%s detect_lang_offset=%s",
            audio_file.filename,
            video_file,
            detect_lang_length,
            detect_lang_offset,
        )

        result = engine.detect_language_file(
            file_bytes=audio,
            filename=audio_file.filename,
            detect_lang_length=detect_lang_length,
            detect_lang_offset=detect_lang_offset,
        )

        _track_request("/detect-language", 200)
        return DetectedLanguage(**result.to_dict())
    except Exception as e:
        _track_request("/detect-language", 500)
        raise HTTPException(status_code=500, detail=f"ASR error: {e}") from e
    finally:
        # Always release the concurrency slot
        release_engine()


@router.get("/models", response_model=ListModelsResponse)
def list_models() -> Any:
    """
    OpenAI-compatible models listing. Delegates to the engine so other
    API mocks (or future services) can reuse the same definition.

    Note: This endpoint uses get_audio_engine() directly (not acquire_engine())
    because it's just reading model metadata, not performing inference.
    """
    return ListModelsResponse(**_engine_for_listing.list_models())


@router.get("/stats")
def get_stats() -> Any:
    """
    Get statistics about Bazarr endpoint usage.

    Returns request counts by endpoint and status code, along with
    success/error rates. Useful for monitoring endpoint health.
    """
    # Convert defaultdict to regular dict for JSON serialization
    stats_output = {
        "started_at": _stats["started_at"],
        "total_requests": _stats["total_requests"],
        "total_success": _stats["total_success"],
        "total_errors": _stats["total_errors"],
        "success_rate": (
            round(_stats["total_success"] / _stats["total_requests"] * 100, 2) if _stats["total_requests"] > 0 else 0.0
        ),
        "endpoints": {endpoint: dict(counts) for endpoint, counts in _stats["endpoints"].items()},
    }

    return JSONResponse(content=stats_output)
