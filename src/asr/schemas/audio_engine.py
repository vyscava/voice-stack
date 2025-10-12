from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelData(BaseModel):
    """
    Represents a single model entry in an OpenAI-compatible models list.

    This mirrors OpenAI’s /v1/models response while allowing extensions:
      - 'parent' denotes model family (e.g. whisper-1 → whisper-1-base)
      - 'supported_tasks' enumerates available endpoints (e.g. transcriptions, translations)
      - 'active' is an internal convenience field indicating the engine’s current model
    """

    id: str = Field(..., description="Unique identifier for the model (e.g. 'whisper-1-large-v3').")
    object: Literal["model"] = Field("model", description="Type discriminator; always 'model'.")
    owned_by: str = Field(..., description="Identifier for the model’s owner (e.g. 'voice-stack').")
    parent: str | None = Field(
        None,
        description="If this model is a variant, indicates the parent model (e.g. 'whisper-1').",
    )
    supported_tasks: list[str] | None = Field(
        default=None,
        description="List of tasks the model supports, such as 'transcriptions' or 'translations'.",
    )
    active: bool | None = Field(
        default=False,
        description="Indicates whether this variant is the currently active model in the engine.",
    )


class ListModelsResponse(BaseModel):
    """
    OpenAI-compatible response for /v1/models.
    """

    object: Literal["list"] = Field("list", description="Type discriminator; always 'list'.")
    data: list[ModelData] = Field(..., description="List of available model entries.")
