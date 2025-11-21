from pydantic import HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    plex_base_url: HttpUrl
    plex_token: str
    plex_library_names: List[str] = ["Movies", "TV Shows"]
    tmdb_api_key: str | None = None
    letterboxd_session: str | None = None
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False)

    @field_validator("plex_token")
    @classmethod
    def ensure_token(cls, value: str) -> str:
        if not value:
            raise ValueError("PLEX_TOKEN must be set")
        return value


settings = Settings()
