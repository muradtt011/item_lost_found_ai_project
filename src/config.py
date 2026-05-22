
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- AI provider ---
    llm_provider: str = Field(default="offline", description="VLM provider name")
    llm_model: str = Field(default="", description="VLM model identifier")
    embedding_provider: str = Field(default="offline", description="Embedding provider")
    embedding_model: str = Field(default="", description="Embedding model identifier")

    # --- API keys (never hardcoded) ---
    google_api_key: str = Field(default="", description="Google / Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # --- Storage ---
    database_url: str = Field(
        default="sqlite+aiosqlite:///./lostfound.db",
        description="Async DB URL; use postgresql+asyncpg:// for Postgres",
    )
    image_storage_dir: Path = Field(
        default=Path("./storage/images"),
        description="Directory where uploaded image blobs are stored",
    )

    # --- Server ---
    http_port: int = Field(default=8000, ge=1, le=65535)

    # --- Validation ---
    max_image_size_mb: float = Field(default=5.0, gt=0)

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # --- Concurrency ---
    ai_concurrency_limit: int = Field(
        default=4, gt=0, description="Max parallel AI calls via semaphore"
    )
    ai_timeout_seconds: float = Field(default=30.0, gt=0)
    ai_max_retries: int = Field(default=3, ge=0)

    @field_validator("image_storage_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(str(v))

    @property
    def max_image_size_bytes(self) -> int:
        return int(self.max_image_size_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    cfg = settings or get_settings()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
