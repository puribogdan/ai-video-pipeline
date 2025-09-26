# config.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root and look for .env there (and one level up as a fallback)
ROOT = Path(__file__).resolve().parent
ENV_FILES = [ROOT / ".env", ROOT.parent / ".env"]

class Settings(BaseSettings):
    # Tell pydantic-settings where to read env vars from
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=[str(p) for p in ENV_FILES],
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown vars in .env
    )

    # --- API keys (optional: scripts will validate when needed) ---
    DEEPGRAM_API_KEY: Optional[str]      = Field(default=None)
    OPENAI_API_KEY: Optional[str]        = Field(default=None)
    CLAUDE_API_KEY: Optional[str]        = Field(default=None)
    REPLICATE_API_TOKEN: Optional[str]   = Field(default=None)
    LUMAAI_API_KEY: Optional[str]        = Field(default=None)

    # --- Defaults for your pipeline (you can override in .env if you want) ---
    IMAGE_PROVIDER: str = Field(default="replicate:flux-schnell")
    VIDEO_PROVIDER: str = Field(default="replicate:seedance-1-pro")

    # Useful paths if you want them elsewhere
    PROJECT_ROOT: Path = ROOT

settings = Settings()
