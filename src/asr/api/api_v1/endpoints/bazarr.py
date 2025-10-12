from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from asr.schemas.bazarr import BazarrTranscription
from core.logging import logger_asr as logger

router = APIRouter()


@router.post("/transcribe", response_model=BazarrTranscription)
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Query(default=None),
    translate: bool = Query(default=False),
    word_timestamps: bool = Query(default=False),
) -> Any:
    """
    Returns segments suitable for creating subtitles in Bazarr workflows.
    TODO: Confirm exact shape Bazarr expects; this is a sane default:
      { language: "en", segments: [ {start, end, text}, ... ] }
    """
    try:
        # audio = await file.read()
        # out = engine.transcribe_bytes(
        #     audio,
        #     request_language=language,
        #     task="translate" if translate else "transcribe",
        #     word_timestamps=word_timestamps,
        # )
        # return BazarrTranscription(language=out["language"], segments=out["segments"])
        logger.info("Bazarr Transcription")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR error: {e}") from e


@router.post("/srt")
async def transcribe_srt(
    file: UploadFile = File(...),
    language: str | None = Query(default=None),
    translate: bool = Query(default=False),
) -> Any:
    """
    Direct SRT output for Bazarr. Content-Type: text/plain
    """
    try:
        # audio = await file.read()
        # out = engine.transcribe_bytes(
        #     audio,
        #     request_language=language,
        #     task="translate" if translate else "transcribe",
        # )
        # srt = segments_to_srt(out["segments"])
        # return Response(content=srt, media_type="text/plain; charset=utf-8")
        logger.info("Bazarr Transcribe SRT")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR error: {e}") from e
