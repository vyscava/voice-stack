from fastapi import APIRouter

from gateway.api.api_v1.endpoints import voice

api_router = APIRouter()
api_router.include_router(voice.router, prefix="/v1/voice", tags=["Voice gateway"])
