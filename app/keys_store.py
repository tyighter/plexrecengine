from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import yaml

KEYS_PATH = Path("/mnt/disks/ssd/plexrec/app/keys.yml")


def load_keys() -> Dict[str, object]:
    """Load saved credentials from the host-accessible YAML file."""
    if not KEYS_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(KEYS_PATH.read_text()) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if v is not None}


def persist_keys(**updates: object):
    """Persist credentials to the host-accessible YAML file."""
    KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = load_keys()
    existing.update({k: v for k, v in updates.items() if v})

    KEYS_PATH.write_text(yaml.safe_dump(existing, sort_keys=True))

    env_mappings = {
        "plex_base_url": "PLEX_BASE_URL",
        "plex_token": "PLEX_TOKEN",
        "plex_library_names": "PLEX_LIBRARY_NAMES",
        "plex_movie_library": "PLEX_MOVIE_LIBRARY",
        "plex_show_library": "PLEX_SHOW_LIBRARY",
        "tmdb_api_key": "TMDB_API_KEY",
    }

    for key, env_name in env_mappings.items():
        value = existing.get(key)
        if isinstance(value, list):
            os.environ[env_name] = ",".join(map(str, value))
        elif value:
            os.environ[env_name] = str(value)
