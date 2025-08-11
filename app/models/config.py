from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core
    environment: Literal["development", "staging", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = Field(default="INFO")
    data_dir: str = Field(default="./data")

    # Embeddings / Generation
    embedding_backend: Literal["openai", "cohere", "sentence-transformers", "hash"] = Field(
        default="hash"
    )
    embedding_model: str = Field(default="hash")
    generation_backend: Literal["openai", "anthropic", "ollama"] = Field(default="openai")
    generation_model: str = Field(default="gpt-4o")

    # Retrieval
    retriever_backend: Literal["faiss", "pinecone", "weaviate", "chroma"] = Field(default="chroma")
    index_dir: str = Field(default="./data/index")
    uploads_dir: str = Field(default="./data/uploads")

    # Provider Keys
    openai_api_key: str | None = None
    cohere_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Security
    allowed_origins: str = Field(default="*")
    admin_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


