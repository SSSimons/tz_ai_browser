from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-5.4-mini", validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = Field(
        default="low", validation_alias="AGENT_REASONING_EFFORT"
    )
    max_steps: int = Field(default=100, ge=1, le=1000, validation_alias="AGENT_MAX_STEPS")
    max_api_calls: int = Field(default=0, ge=0, validation_alias="AGENT_MAX_API_CALLS")
    max_output_tokens: int = Field(default=5000, ge=128, validation_alias="AGENT_MAX_OUTPUT_TOKENS")
    api_max_retries: int = Field(default=0, ge=0, le=3, validation_alias="AGENT_API_MAX_RETRIES")
    context_chars: int = Field(default=32000, ge=8000, validation_alias="AGENT_CONTEXT_CHARS")
    record_actions: bool = Field(default=False, validation_alias="AGENT_RECORD_ACTIONS")
    knowledge_path: Path = Field(
        default=Path(".knowledge/facts.json"), validation_alias="AGENT_KNOWLEDGE_PATH"
    )
    tool_timeout_seconds: int = Field(
        default=45, ge=5, validation_alias="AGENT_TOOL_TIMEOUT_SECONDS"
    )
    headless: bool = Field(default=False, validation_alias="AGENT_HEADLESS")
    profile_dir: Path = Field(
        default=Path(".browser-profile"), validation_alias="AGENT_PROFILE_DIR"
    )
    browser_executable: str | None = Field(
        default=None, validation_alias="AGENT_BROWSER_EXECUTABLE"
    )
    cdp_url: str | None = Field(default=None, validation_alias="AGENT_CDP_URL")
    artifacts_dir: Path = Field(default=Path("artifacts"), validation_alias="AGENT_ARTIFACTS_DIR")
    run_dir: Path | None = Field(default=None, validation_alias="AGENT_RUN_DIR")
