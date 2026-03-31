from typing import Any

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    detail: str = Field(..., description='Error message')