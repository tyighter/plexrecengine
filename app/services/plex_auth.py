from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import set_key
from plexapi.myplex import MyPlexAccount, MyPlexPinLogin
from plexapi.server import PlexServer

from app.config import settings

_PENDING_LOGINS: Dict[str, tuple[MyPlexPinLogin, datetime]] = {}
_LOGIN_TIMEOUT = timedelta(minutes=10)
ENV_PATH = Path(".env")


class PlexLoginStatus:
    def __init__(
        self,
        status: str,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        server_name: Optional[str] = None,
        libraries: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.status = status
        self.token = token
        self.base_url = base_url
        self.server_name = server_name
        self.libraries = libraries

    def dict(self):
        return {
            "status": self.status,
            "token": self.token,
            "baseUrl": self.base_url,
            "serverName": self.server_name,
            "libraries": self.libraries,
        }


def start_login() -> dict:
    pin_login = MyPlexPinLogin(oauth=True)
    _PENDING_LOGINS[pin_login._id] = (pin_login, datetime.utcnow())
    return {
        "pinId": pin_login._id,
        "authUrl": pin_login.oauthUrl(),
        "expiresIn": int(_LOGIN_TIMEOUT.total_seconds()),
    }


def _remove_expired(pin_id: str):
    _PENDING_LOGINS.pop(pin_id, None)


def _connect_server(account: MyPlexAccount) -> Optional[PlexServer]:
    for resource in account.resources():
        if "server" not in resource.provides:
            continue
        try:
            return resource.connect()
        except Exception:
            continue
    return None


def _collect_library_names(server: PlexServer) -> Dict[str, List[str]]:
    names: Dict[str, List[str]] = {"movie": [], "show": []}
    for section in server.library.sections():
        section_type = getattr(section, "TYPE", None)
        if section_type in names:
            names[section_type].append(section.title)
    if not names["movie"]:
        names["movie"] = ["Movies"]
    if not names["show"]:
        names["show"] = ["TV Shows"]
    return names


def _persist_settings(base_url: str, token: str, movie_library: str, show_library: str):
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "PLEX_BASE_URL", base_url)
    set_key(str(ENV_PATH), "PLEX_TOKEN", token)
    set_key(str(ENV_PATH), "PLEX_LIBRARY_NAMES", ",".join([movie_library, show_library]))
    set_key(str(ENV_PATH), "PLEX_MOVIE_LIBRARY", movie_library)
    set_key(str(ENV_PATH), "PLEX_SHOW_LIBRARY", show_library)
    settings.plex_base_url = base_url
    settings.plex_token = token
    settings.plex_library_names = [movie_library, show_library]
    settings.plex_movie_library = movie_library
    settings.plex_show_library = show_library


def _validate_library_choice(library_name: str, available: Dict[str, List[str]], media_type: str) -> str:
    cleaned = (library_name or "").strip()
    if cleaned and cleaned in available.get(media_type, []):
        return cleaned
    defaults = {"movie": "Movies", "show": "TV Shows"}
    return available.get(media_type, [defaults[media_type]])[0]


def list_available_libraries() -> Dict[str, List[str]]:
    if not settings.is_plex_configured:
        return {"movie": [], "show": []}
    try:
        server = PlexServer(str(settings.plex_base_url), settings.plex_token)
        return _collect_library_names(server)
    except Exception as exc:
        raise RuntimeError("Unable to load libraries from Plex") from exc


def save_library_preferences(movie_library: str, show_library: str):
    if not settings.is_plex_configured:
        raise RuntimeError("Plex must be configured before updating libraries")

    available = list_available_libraries()
    movie = _validate_library_choice(movie_library, available, media_type="movie")
    show = _validate_library_choice(show_library, available, media_type="show")

    _persist_settings(str(settings.plex_base_url), str(settings.plex_token), movie, show)
    return {"movie": movie, "show": show}


def check_login(pin_id: str) -> PlexLoginStatus:
    entry = _PENDING_LOGINS.get(pin_id)
    if not entry:
        return PlexLoginStatus("invalid")

    pin_login, created_at = entry
    if datetime.utcnow() - created_at > _LOGIN_TIMEOUT:
        _remove_expired(pin_id)
        return PlexLoginStatus("expired")

    try:
        pin_login.checkLogin()
    except Exception:
        _remove_expired(pin_id)
        return PlexLoginStatus("error")

    if pin_login.expired:
        _remove_expired(pin_id)
        return PlexLoginStatus("expired")

    if not pin_login.token:
        return PlexLoginStatus("pending")

    account = MyPlexAccount(token=pin_login.token)
    server = _connect_server(account)
    if not server:
        _remove_expired(pin_id)
        return PlexLoginStatus("error")

    libraries = _collect_library_names(server)
    movie_library = _validate_library_choice(settings.plex_movie_library, libraries, media_type="movie")
    show_library = _validate_library_choice(settings.plex_show_library, libraries, media_type="show")
    base_url = server.url("") if callable(getattr(server, "url", None)) else str(server.url)
    _persist_settings(base_url, pin_login.token, movie_library, show_library)
    _remove_expired(pin_id)
    return PlexLoginStatus(
        "authorized",
        token=pin_login.token,
        base_url=base_url,
        server_name=getattr(server, "friendlyName", None),
        libraries=libraries,
    )
