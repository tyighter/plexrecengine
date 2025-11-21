from typing import List

from pydantic import HttpUrl, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    plex_base_url: HttpUrl | None = None
    plex_token: str | None = None
    plex_library_names: List[str] | None = None
    tmdb_api_key: str | None = None
    letterboxd_session: str | None = None
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        env_parse_none_str="",
    )

    @field_validator("plex_library_names", mode="before")
    @classmethod
    def split_library_names(cls, value):
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        return value

    @model_validator(mode="after")
    def set_default_library_names(self):
        if not self.plex_library_names:
            self.plex_library_names = ["Movies", "TV Shows"]
        return self

    @computed_field
    @property
    def is_plex_configured(self) -> bool:
        return bool(self.plex_base_url and self.plex_token)


settings = Settings()
