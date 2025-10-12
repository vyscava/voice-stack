from fastapi import APIRouter

from asr.api.api_v1.endpoints import bazarr, openai

api_router = APIRouter()
api_router.include_router(openai.router, prefix="/v1", tags=["OpenAI-compatible ASR"])
api_router.include_router(bazarr.router, prefix="/bazarr", tags=["Bazarr-compatible ASR"])
