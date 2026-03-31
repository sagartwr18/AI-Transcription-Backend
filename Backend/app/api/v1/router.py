from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.transcription import router as transcription_router

api_router = APIRouter()
api_router.include_router(health_router, tags=['Health'])
api_router.include_router(transcription_router, prefix='/transcription', tags=['Transcription'])