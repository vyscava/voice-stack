"""The gateway's HTTP surface: one utterance in, one spoken reply out."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, Response

from core.logging import logger_gateway as logger
from core.settings import get_settings
from gateway.deps import build_agent, build_policy, build_speaker, build_transcriber
from gateway.exchange import Outcome, run_exchange

router = APIRouter()
settings = get_settings()

_MEDIA_TYPES = {
    "opus": "audio/ogg",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "application/octet-stream",
}

# Carried on the response so a caller can tell a real answer from a spoken
# apology WITHOUT parsing the audio. Without these an exchange that failed and
# an exchange that worked are both "200 with some audio".
_HEADER_OUTCOME = "X-Voice-Outcome"
_HEADER_DEGRADED = "X-Voice-Degraded"
_HEADER_TRANSCRIPT = "X-Voice-Transcript"


@router.post("/exchange", summary="Speak to an agent and hear its reply")
async def exchange(file: UploadFile = File(...)) -> Response:
    """audio in -> ASR -> agent -> TTS -> audio out.

    Always returns 200 with audio when it can, including for failures, because
    the caller is a person holding a microphone. An HTTP error code is not
    something they can hear. The outcome is reported in headers instead.
    """
    audio = await file.read()
    filename = file.filename or "utterance.wav"

    result = await run_exchange(
        audio,
        filename=filename,
        transcriber=build_transcriber(),
        speaker=build_speaker(),
        agent=build_agent(),
        policy=build_policy(),
    )

    log = logger.info if result.outcome is Outcome.OK else logger.warning
    log(
        f"exchange outcome={result.outcome.value} "
        f"transcript={(result.transcript or '')[:80]!r} "
        f"detail={result.detail or '-'}"
    )

    headers = {
        _HEADER_OUTCOME: result.outcome.value,
        _HEADER_DEGRADED: str(result.degraded).lower(),
        # Header values must be latin-1 encodable; a Portuguese transcript is
        # not. Encode rather than crash on the way out.
        _HEADER_TRANSCRIPT: (result.transcript or "").encode("unicode_escape").decode("ascii")[:512],
    }

    if result.audio is None:
        # TTS is down, or failed on top of an earlier failure. The answer still
        # exists, so return it as text rather than discarding it.
        return JSONResponse(
            status_code=200,
            content={
                "outcome": result.outcome.value,
                "degraded": result.degraded,
                "transcript": result.transcript,
                "reply_text": result.reply_text,
                "detail": result.detail,
                "audio": None,
            },
            headers=headers,
        )

    return Response(
        content=result.audio,
        media_type=_MEDIA_TYPES.get(settings.GATEWAY_AUDIO_FORMAT, "application/octet-stream"),
        headers=headers,
    )


@router.post("/transcribe-only", summary="Hear an utterance without asking an agent")
async def transcribe_only(file: UploadFile = File(...)) -> JSONResponse:
    """Useful for checking the microphone path without spending an agent turn."""
    audio = await file.read()
    transcriber = build_transcriber()
    try:
        text = await transcriber.transcribe(audio, filename=file.filename or "utterance.wav")
    except Exception as exc:
        logger.warning(f"transcribe-only failed: {type(exc).__name__}: {exc}")
        return JSONResponse(status_code=502, content={"error": f"{type(exc).__name__}: {exc}"})
    return JSONResponse({"text": text})
