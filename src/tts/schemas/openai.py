from __future__ import annotations

from pydantic import BaseModel, Field

from tts.schemas.audio_engine import AudioFormat, StreamFormat


class OpenAICreateSpeechRequest(BaseModel):
    model: str = Field(default="TTS (XTTS)", description="e.g., gpt-4o-mini-tts or similar")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(..., description="Voice name, e.g., alloy, verse, shimmer")
    response_format: AudioFormat = Field(
        AudioFormat.MP3, description="The format to audio in. Supported formats are mp3, opus, aac, flac, wav, and pcm"
    )
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: StreamFormat | None = Field(default=StreamFormat.AUDIO, description="File or Stream response..")
