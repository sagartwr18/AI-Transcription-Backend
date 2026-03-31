from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / '.env')


class Settings(BaseSettings):
    PROJECT_NAME: str = 'AI Transcription Backend'
    VERSION: str = '1.0.0'
    API_V1_PREFIX: str = '/api/v1'
    CORS_ORIGINS: str = 'http://localhost:5173,http://127.0.0.1:5173'
    CORS_ALLOW_CREDENTIALS: bool = True
    ASSEMBLYAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    DEEPGRAM_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / '.env', extra='ignore')

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(',') if origin.strip()]


settings = Settings()
