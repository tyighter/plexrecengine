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
        libraries: Optional[List[str]] = None,
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


def _collect_library_names(server: PlexServer) -> List[str]:
    names: List[str] = []
    for section in server.library.sections():
        if getattr(section, "TYPE", None) in {"movie", "show"}:
            names.append(section.title)
    return names or ["Movies", "TV Shows"]


def _persist_settings(base_url: str, token: str, libraries: List[str]):
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "PLEX_BASE_URL", base_url)
    set_key(str(ENV_PATH), "PLEX_TOKEN", token)
    set_key(str(ENV_PATH), "PLEX_LIBRARY_NAMES", ",".join(libraries))
    settings.plex_base_url = base_url
    settings.plex_token = token
    settings.plex_library_names = libraries


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
    _persist_settings(server.url, pin_login.token, libraries)
    _remove_expired(pin_id)
    return PlexLoginStatus(
        "authorized",
        token=pin_login.token,
        base_url=server.url,
        server_name=getattr(server, "friendlyName", None),
        libraries=libraries,
    )
