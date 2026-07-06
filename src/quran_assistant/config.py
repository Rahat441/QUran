"""Application configuration loaded from environment variables and .env."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Environment variables use the QRA_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="QRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    chroma_path: Path = Path("data/chroma")
    collection_name: str = "quran_verses"
    top_k: int = Field(default=8, ge=1, le=50)
    candidate_k: int = Field(default=24, ge=1, le=200)
    semantic_weight: float = Field(default=0.75, ge=0, le=1)
    minimum_relevance: float = Field(default=0.18, ge=0, le=1)


def get_settings() -> Settings:
    """Build settings at the point of use so tests can override the environment."""

    return Settings()
