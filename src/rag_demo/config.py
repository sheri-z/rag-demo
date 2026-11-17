"""Settings loaded from environment variables or the .env file."""

from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    openai_api_key: SecretStr
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_openai() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key.get_secret_value())
