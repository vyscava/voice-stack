from __future__ import annotations

from fastapi import Form
from pydantic import BaseModel, Field


class OpenAITranscriptionRequest(BaseModel):
    """Input model matching OpenAI-compatible transcription fields."""

    model: str = Field("whisper-1", description="Model name; kept for OpenAI compatibility.")
    language: str | None = Field(None, description="Optional BCP-47 language hint (e.g. 'en', 'pt').")
    temperature: float | None = Field(None, description="Sampling temperature; 0.0 for deterministic output.")
    response_format: str = Field("json", description="Response format: 'json' or 'text'.")
    prompt: str | None = Field(None, description="Prompt text; ignored for Whisper.")
    timestamp_granularities: str | None = Field(None, description="Ignored (OpenAI placeholder).")

    @classmethod
    def as_form(
        cls,
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        temperature: float | None = Form(None),
        response_format: str = Form("json"),
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

    task: str | None = Field(None)
    beam_size: int | None = Field(None)
    best_of: int | None = Field(None)
    word_timestamps: bool | None = Field(None)
    vad: bool | None = Field(None)

    @classmethod
    def as_form(
        cls,
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        temperature: float | None = Form(None),
        response_format: str = Form("json"),
        prompt: str | None = Form(None),
        timestamp_granularities: str | None = Form(None),
        task: str | None = Form(None),
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
    response_format: str = Field("json", description="Response format: 'json' or 'text'.")
    prompt: str | None = Field(None, description="Prompt text; ignored for Whisper.")

    @classmethod
    def as_form(
        cls,
        model: str = Form("whisper-1"),
        temperature: float | None = Form(None),
        response_format: str = Form("json"),
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

    language_probability: float | None = Field(default=None, description="Confidence for detected language (0..1).")
    duration: float | None = Field(default=None, description="Approx. audio duration in seconds.")
    processing_ms: int | None = Field(default=None, description="End-to-end processing time in milliseconds.")
    asr_ms: int | None = Field(default=None, description="Time spent in ASR model inference (ms).")
    vad_used: bool | None = Field(default=None, description="Whether VAD preprocessing was applied.")
    model: str | None = Field(default=None, description="Underlying ASR model identifier.")
    timings: dict[str, float] | None = Field(default=None, description="Optional timing breakdown by stage.")


# OpenAI-style minimal response for /transcriptions and /translations
class OpenAITranscription(BaseModel):
    """OpenAI-compatible minimal payload."""

    text: str
