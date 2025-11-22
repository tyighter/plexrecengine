import json
import os
from pathlib import Path
from typing import List, Optional

from pydantic import HttpUrl, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.keys_store import load_keys

CONFIG_DIR = Path(".data")
CONFIG_PATH = CONFIG_DIR / "config.json"


def _load_saved_config():
    yaml_config = load_keys()
    json_config = {}
    if CONFIG_PATH.exists():
        try:
            json_config = json.loads(CONFIG_PATH.read_text())
        except Exception:
            json_config = {}

    data = {**yaml_config, **json_config}

    def _set_env(name: str, value: object):
        if value is None:
            return
        if isinstance(value, list):
            env_value = ",".join(map(str, value))
        else:
            env_value = str(value)
        os.environ.setdefault(name, env_value)

    for key, value in data.items():
        _set_env(key, value)
        upper_key = key.upper()
        if upper_key != key:
            _set_env(upper_key, value)


def save_config(values: dict[str, object]):
    CONFIG_DIR.mkdir(exist_ok=True)
    existing: dict[str, object] = {}
    if CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text())
        except Exception:
            existing = {}

    existing.update(values)
    CONFIG_PATH.write_text(json.dumps(existing, indent=2))


_load_saved_config()


class Settings(BaseSettings):
    plex_base_url: HttpUrl | None = None
    plex_token: str | None = None
    plex_library_names: List[str] | None = None
    plex_movie_library: Optional[str] = None
    plex_show_library: Optional[str] = None
    plex_user_id: str | None = None
    tautulli_base_url: HttpUrl | None = None
    tautulli_api_key: str | None = None
    tautulli_user_id: str | None = None
    tmdb_api_key: str | None = None
    letterboxd_session: str | None = None
    related_pool_limit: int = 20
    dashboard_timeout_seconds: float = 10.0
    recent_activity_timeout_seconds: float = 10.0
    recommendation_build_timeout_seconds: float = 120.0
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
        if not self.plex_movie_library or not self.plex_show_library:
            libraries = self.plex_library_names or []
            if not self.plex_movie_library and libraries:
                self.plex_movie_library = libraries[0]
            if not self.plex_show_library and len(libraries) > 1:
                self.plex_show_library = libraries[1]

        self.plex_movie_library = self.plex_movie_library or "Movies"
        self.plex_show_library = self.plex_show_library or "TV Shows"
        self.plex_library_names = [self.plex_movie_library, self.plex_show_library]
        return self

    @computed_field
    @property
    def is_plex_configured(self) -> bool:
        return bool(self.plex_base_url and self.plex_token)

    @computed_field
    @property
    def is_tautulli_configured(self) -> bool:
        return bool(self.tautulli_base_url and self.tautulli_api_key)


settings = Settings()
