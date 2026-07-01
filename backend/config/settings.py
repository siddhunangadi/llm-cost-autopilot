from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "test", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: str = Field(default="sqlite:///./llm_cost_autopilot.db", min_length=1)
    models_yaml_path: str = Field(default="backend/config/models.yaml", min_length=1)

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
