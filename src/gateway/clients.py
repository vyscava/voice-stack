"""HTTP clients for the ASR and TTS services.

The gateway does not import the engines. It calls the same public endpoints any
other consumer uses, which keeps it a thin process, lets the heavy services
scale and restart independently, and means the gateway can be exercised against
mocks in a test rather than against a 1.8 GB model.
"""

from __future__ import annotations

import httpx

from core.logging import logger_gateway as logger


class AsrHttpClient:
    """Speech to text, via the ASR service's OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, *, timeout_s: float = 120.0, language: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._language = language

    async def transcribe(self, audio: bytes, *, filename: str) -> str:
        files = {"file": (filename, audio, "application/octet-stream")}
        data: dict[str, str] = {"model": "whisper-1"}
        if self._language:
            data["language"] = self._language

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/v1/audio/transcriptions", files=files, data=data)
            response.raise_for_status()
            return _extract_transcript(response)


class TtsHttpClient:
    """Text to speech, via the TTS service's OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        voice: str,
        audio_format: str = "opus",
        timeout_s: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._audio_format = audio_format
        self._timeout_s = timeout_s

    async def speak(self, text: str) -> bytes:
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": self._voice,
            "response_format": self._audio_format,
        }

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/v1/audio/speech", json=payload)
            response.raise_for_status()
            audio = response.content

        # An empty 200 is the failure mode worth naming: the caller would get a
        # valid-looking response containing no sound, which is indistinguishable
        # from the gateway having broken.
        if not audio:
            raise RuntimeError("TTS returned an empty body with a success status")
        return audio


def _extract_transcript(response: httpx.Response) -> str:
    """Pull the text out, whether the service answered JSON or plain text.

    The endpoint's shape depends on the requested response_format, so accept
    both rather than coupling the gateway to one of them.
    """
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        body = response.json()
        if isinstance(body, dict):
            text = body.get("text")
            if isinstance(text, str):
                return text
            logger.warning("ASR JSON response had no usable 'text' field")
            return ""
        if isinstance(body, str):
            return body
        return ""
    return response.text
