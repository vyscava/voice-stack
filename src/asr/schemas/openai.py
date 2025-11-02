from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import Form
from pydantic import BaseModel, Field, field_validator

from asr.schemas.audio_engine import Task


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"


def _empty_to_none(v: str | None) -> str | None:
    # Normalize browser-submitted empty strings from multipart/form-data
    return None if v == "" else v


def _parse_optional_float(v: Any) -> float | None:
    v = _empty_to_none(v)
    if v is None:
        return None
    if isinstance(v, int | float):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None  # or raise ValueError(...) if you prefer strictness


def _parse_optional_int(v: Any) -> int | None:
    v = _empty_to_none(v)
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(str(v))
    except Exception:
        return None


def _parse_optional_bool(v: Any) -> bool | None:
    v = _empty_to_none(v)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


class OpenAITranscriptionRequest(BaseModel):
    """Input model matching OpenAI-compatible transcription fields."""

    model: str = Field("whisper-1", description="Model name; kept for OpenAI compatibility.")
    language: str | None = Field(None, description="Optional BCP-47 language hint (e.g. 'en', 'pt').")
    temperature: float | None = Field(None, description="Sampling temperature; 0.0 for deterministic output.")
    response_format: ResponseFormat = Field(ResponseFormat.JSON, description="Response format: 'json' or 'text'.")
    prompt: str | None = Field(None, description="Prompt text; ignored for Whisper.")
    timestamp_granularities: str | None = Field(None, description="Ignored (OpenAI placeholder).")

    # Coerce "" -> None for optional fields that arrive as strings
    @field_validator("language", "prompt", "timestamp_granularities", mode="before")
    @classmethod
    def _coerce_empty_to_none(cls, v: Any) -> Any:
        return _empty_to_none(v)

    @field_validator("temperature", mode="before")
    @classmethod
    def _coerce_temp(cls, v: Any) -> Any:
        return _parse_optional_float(v)

    @classmethod
    def as_form(
        cls,
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        temperature: float | None = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        prompt: str | None = Form(None),
        timestamp_granularities: str | None = Form(None),
    ) -> OpenAITranscriptionRequest:
        return cls(
            model=model,
            language=language,
            temperature=temperature,
            response_format=response_format,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )


class OpenAITranscriptionVerboseRequest(OpenAITranscriptionRequest):
    """Extended model for verbose mode."""

    task: Task | None = Field(None)
    beam_size: int | None = Field(None)
    best_of: int | None = Field(None)
    word_timestamps: bool | None = Field(None)
    vad: bool | None = Field(None)

    # Coerce "" -> None for optional numerics/bools too
    @field_validator("task", mode="before")
    @classmethod
    def _coerce_empty_verbose(cls, v: Any) -> Any:
        return _empty_to_none(v)

    @field_validator("beam_size", "best_of", mode="before")
    @classmethod
    def _coerce_ints(cls, v: Any) -> Any:
        return _parse_optional_int(v)

    @field_validator("word_timestamps", "vad", mode="before")
    @classmethod
    def _coerce_bools(cls, v: Any) -> Any:
        return _parse_optional_bool(v)

    @classmethod
    def as_form(
        cls,
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        temperature: float | None = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        prompt: str | None = Form(None),
        timestamp_granularities: str | None = Form(None),
        task: Task | None = Form(None),
        beam_size: int | None = Form(None),
        best_of: int | None = Form(None),
        word_timestamps: bool | None = Form(None),
        vad: bool | None = Form(None),
    ) -> OpenAITranscriptionVerboseRequest:
        return cls(
            model=model,
            language=language,
            temperature=temperature,
            response_format=response_format,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
            task=task,
            beam_size=beam_size,
            best_of=best_of,
            word_timestamps=word_timestamps,
            vad=vad,
        )


class OpenAITranslationsRequest(BaseModel):
    """Input model matching OpenAI-compatible translations fields."""

    model: str = Field("whisper-1", description="Model name; kept for OpenAI compatibility.")
    temperature: float | None = Field(None, description="Sampling temperature; 0.0 for deterministic output.")
    response_format: ResponseFormat = Field(ResponseFormat.JSON, description="Response format: 'json' or 'text'.")
    prompt: str | None = Field(None, description="Prompt text; ignored for Whisper.")

    @classmethod
    def as_form(
        cls,
        model: str = Form("whisper-1"),
        temperature: float | None = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        prompt: str | None = Form(None),
    ) -> OpenAITranslationsRequest:
        return cls(
            model=model,
            temperature=temperature,
            response_format=response_format,
            prompt=prompt,
        )


class Segment(BaseModel):
    """One continuous chunk of recognized speech."""

    start: float = Field(..., description="Segment start time in seconds.")
    end: float = Field(..., description="Segment end time in seconds.")
    text: str = Field(..., description="Recognized text for this segment.")


class TranscribeResponse(BaseModel):
    """
    Compact response for internal/verbose endpoints.
    """

    text: str = Field(..., description="Full concatenated transcript.")
    segments: list[Segment] = Field(default_factory=list, description="Per-segment results.")
    language: str | None = Field(
        default=None,
        description="Detected or requested BCP-47-ish language code (e.g., 'en', 'pt', 'zh-cn').",
    )


class TranscribeVerboseResponse(TranscribeResponse):
    """
    Rich response for debugging/analytics (your '/verbose' route).
    Extend freely as your engine returns more metadata.
    """

    language_name: str | None = Field(
        default=None,
        description="Detected or requested BCP-47-ish language name (e.g., 'English', 'Portuguese').",
    )
    language_probability: float | None = Field(default=None, description="Confidence for detected language (0..1).")
    duration_input_s: float | None = Field(default=None, description="Approx. audio duration in seconds.")
    duration_after_vad_s: float | None = Field(default=None, description="Approx. audio duration after VAD in seconds.")
    processing_ms: int | None = Field(default=None, description="End-to-end processing time in milliseconds.")
    asr_ms: int | None = Field(default=None, description="Time spent in ASR model inference (ms).")
    vad_used: bool | None = Field(default=None, description="Whether VAD preprocessing was applied.")
    model: str | None = Field(default=None, description="Underlying ASR model identifier.")


# OpenAI-style minimal response for /transcriptions and /translations
class OpenAITranscription(BaseModel):
    """OpenAI-compatible minimal payload."""

    text: str
