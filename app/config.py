# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None

    VOYAGE_API_KEY: str | None = None  # <-- REQUIRED for VoyageAI

    QDRANT_URL: str
    QDRANT_API_KEY: str 
    EMBEDDING_DIM: int = 384

    PREDEFINED_COLLECTION: str = "predefined_context"
    USER_HISTORY_COLLECTION: str = "user_history"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
