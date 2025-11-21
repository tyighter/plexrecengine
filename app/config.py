from pydantic import HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    plex_base_url: HttpUrl
    plex_token: str
    plex_library_names: List[str] | None = None
    tmdb_api_key: str | None = None
    letterboxd_session: str | None = None
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        env_parse_none_str="",
    )

    @field_validator("plex_token")
    @classmethod
    def ensure_token(cls, value: str) -> str:
        if not value:
            raise ValueError("PLEX_TOKEN must be set")
        return value

    @model_validator(mode="after")
    def set_default_library_names(self):
        if not self.plex_library_names:
            self.plex_library_names = ["Movies", "TV Shows"]
        return self


settings = Settings()
