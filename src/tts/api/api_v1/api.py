from fastapi import APIRouter

from tts.api.api_v1.endpoints import openai

api_router = APIRouter()
api_router.include_router(openai.router, prefix="/v1", tags=["OpenAI-compatible TTS"])
