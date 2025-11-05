from __future__ import annotations

import datetime
from enum import Enum

from pydantic import BaseModel, Field


class StreamFormat(str, Enum):
    SSE = "sse"
    AUDIO = "audio"


class AudioFormat(str, Enum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


def return_date_as_unix_ts(*, year: int, month: int, day: int) -> int:
    """
    Return a Date as Unix TimeStamp Integer
    """
    return int(
        (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.datetime(year=year, month=month, day=day, tzinfo=datetime.timezone.utc)
        ).total_seconds()
    )


class ModelResponse(BaseModel):
    id: str = Field(..., description="The model identifier, which can be referenced in the API endpoints.")
    object: str = Field(default="model", description="The object type, which is always 'model'.")
    created: int = Field(
        default=return_date_as_unix_ts(year=1900, month=1, day=1),
        description="The Unix timestamp (in seconds) when the model was created.",
    )
    owned_by: str = Field(default="AllOfUs", description="The organization that owns the model.")


class ModelsResponse(BaseModel):
    object: str = Field(default="list")
    data: list[ModelResponse] = Field(...)


class VoiceResponse(BaseModel):
    id: str = Field(..., description="The voice identifier, which can be referenced in the API endpoints.")
    name: str = Field(..., description="The display name of the voice.")
    object: str = Field(default="voice", description="The object type, which is always 'voice'.")


class VoicesResponse(BaseModel):
    object: str = Field(default="list")
    data: list[VoiceResponse] = Field(...)
