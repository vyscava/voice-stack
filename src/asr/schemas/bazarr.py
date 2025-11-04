from __future__ import annotations

from fastapi import Form
from pydantic import BaseModel, Field

from asr.schemas.audio_engine import Output, Task
from utils.language.language_codes import LanguageCode


class BazarrAsrRequest(BaseModel):
    """Input model matching bazarr-compatible asr fields."""

    encode: bool = Field(default=True, description="IGNORED: Encode audio first through ffmpeg")
    task: Task | None = Field(default=Task.TRANSCRIBE, description="Operation should be transcribe or translate")
    language: LanguageCode | None = Field(default=LanguageCode.UNKNOWN, description="Language of the audio if known")
    initial_prompt: str | None = Field(default=None, description="Prompt text; ignored for Whisper.")
    output: Output | None = Field(default=Output.TXT, description="Output formats: txt|vtt|srt|tsv|json")

    @classmethod
    def as_form(
        cls,
        encode: bool = Form(True),
        task: Task | None = Form(Task.TRANSCRIBE),
        language: LanguageCode | None = Form(LanguageCode.UNKNOWN),
        initial_prompt: str | None = Form(None),
        output: Output | None = Form(Output.TXT),
    ) -> BazarrAsrRequest:
        return cls(encode=encode, task=task, language=language, initial_prompt=initial_prompt, output=output)


class BazarrDetectLanguageRequest(BaseModel):
    """Input model matching bazarr-compatible detect language."""

    encode: bool = Field(default=True, description="IGNORED: Encode audio first through ffmpeg")
    detect_lang_length: int | None = Field(default=None, description="Detect language on X seconds of the file")
    detect_lang_offset: int | None = Field(default=None, description="Start Detect language X seconds into the file")
    video_file: str | None = Field(default=None, description="Original video file path for logging purposes")
