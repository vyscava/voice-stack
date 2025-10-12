from __future__ import annotations

from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    text: str
    segments: list[Segment]
    language: str | None = None


# OpenAI-style response
class OpenAITranscription(BaseModel):
    text: str


# Bazarr-style (adjust if your Bazarr integration expects a different shape)
class BazarrTranscription(BaseModel):
    language: str | None = None
    segments: list[Segment]
