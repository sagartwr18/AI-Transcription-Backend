from fastapi import APIRouter

from app.schemas.common import ApiResponse

router = APIRouter()


@router.get('/health', response_model=ApiResponse)
def health_check() -> ApiResponse:
    return ApiResponse(success=True, message='Backend is running')