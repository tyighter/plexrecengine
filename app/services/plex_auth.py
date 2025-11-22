from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import set_key
from plexapi.myplex import MyPlexAccount, MyPlexPinLogin
from plexapi.server import PlexServer

from app.config import save_config, settings
from app.keys_store import persist_keys
from app.services.plex_logging import LOG_FILE, get_plex_logger

_PENDING_LOGINS: Dict[str, tuple[MyPlexPinLogin, datetime]] = {}
_COMPLETED_LOGINS: Dict[str, tuple["PlexLoginStatus", datetime]] = {}
_LOGIN_TIMEOUT = timedelta(minutes=10)
_COMPLETED_RETENTION = timedelta(minutes=2)
ENV_PATH = Path(".env")

LOGGER = get_plex_logger()
LOGGER.debug("Plex authentication logger initialized", extra={"log_file": str(LOG_FILE)})


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
    LOGGER.debug(
        "Started Plex login flow",
        extra={
            "pin_id": pin_login._id,
            "pending_logins": len(_PENDING_LOGINS),
            "expires_in_seconds": int(_LOGIN_TIMEOUT.total_seconds()),
        },
    )
    return {
        "pinId": pin_login._id,
        "authUrl": pin_login.oauthUrl(),
        "expiresIn": int(_LOGIN_TIMEOUT.total_seconds()),
    }


def _remove_expired(pin_id: str):
    LOGGER.debug(
        "Removing expired or completed login",
        extra={"pin_id": pin_id, "pending_before": len(_PENDING_LOGINS)},
    )
    _PENDING_LOGINS.pop(pin_id, None)
    LOGGER.debug("Login removed", extra={"pin_id": pin_id, "pending_after": len(_PENDING_LOGINS)})


def _store_completed(pin_id: str, status: "PlexLoginStatus"):
    cutoff = datetime.utcnow() - _COMPLETED_RETENTION
    for stale_pin in [p for p, (_, ts) in _COMPLETED_LOGINS.items() if ts < cutoff]:
        _COMPLETED_LOGINS.pop(stale_pin, None)
    _COMPLETED_LOGINS[pin_id] = (status, datetime.utcnow())
    LOGGER.debug(
        "Stored completed login status",
        extra={"pin_id": pin_id, "status": status.status, "cached": len(_COMPLETED_LOGINS)},
    )


def _connect_server(account: MyPlexAccount) -> Optional[PlexServer]:
    LOGGER.debug("Attempting to connect to Plex server resources")
    for resource in account.resources():
        if "server" not in resource.provides:
            continue
        try:
            # Avoid long hangs when Plex servers are unreachable by limiting the
            # per-connection timeout.
            LOGGER.debug(
                "Connecting to Plex resource", extra={"resource": resource.name}
            )
            return resource.connect(timeout=5)
        except Exception:
            LOGGER.exception(
                "Failed to connect to Plex resource", extra={"resource": resource.name}
            )
            continue
    LOGGER.error("No accessible Plex server resources found")
    return None


def _collect_library_names(server: PlexServer) -> Dict[str, List[str]]:
    names: Dict[str, List[str]] = {"movie": [], "show": []}
    for section in server.library.sections():
        section_type = getattr(section, "TYPE", None)
        if section_type in names:
            names[section_type].append(section.title)
            LOGGER.debug(
                "Discovered Plex library section",
                extra={"type": section_type, "title": section.title},
            )
    if not names["movie"]:
        names["movie"] = ["Movies"]
    if not names["show"]:
        names["show"] = ["TV Shows"]
    LOGGER.debug("Collected Plex libraries", extra={"libraries": names})
    return names


def _persist_settings(
    base_url: str,
    token: str,
    movie_library: str,
    show_library: str,
    plex_user_id: str | None = None,
):
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "PLEX_BASE_URL", base_url)
    set_key(str(ENV_PATH), "PLEX_TOKEN", token)
    set_key(str(ENV_PATH), "PLEX_LIBRARY_NAMES", ",".join([movie_library, show_library]))
    set_key(str(ENV_PATH), "PLEX_MOVIE_LIBRARY", movie_library)
    set_key(str(ENV_PATH), "PLEX_SHOW_LIBRARY", show_library)
    set_key(str(ENV_PATH), "PLEX_USER_ID", plex_user_id or "")
    persist_keys(
        plex_base_url=base_url,
        plex_token=token,
        plex_library_names=[movie_library, show_library],
        plex_movie_library=movie_library,
        plex_show_library=show_library,
        plex_user_id=plex_user_id or None,
    )
    save_config(
        {
            "PLEX_BASE_URL": base_url,
            "PLEX_TOKEN": token,
            "PLEX_LIBRARY_NAMES": [movie_library, show_library],
            "PLEX_MOVIE_LIBRARY": movie_library,
            "PLEX_SHOW_LIBRARY": show_library,
            "PLEX_USER_ID": plex_user_id or None,
        }
    )
    settings.plex_base_url = base_url
    settings.plex_token = token
    settings.plex_library_names = [movie_library, show_library]
    settings.plex_movie_library = movie_library
    settings.plex_show_library = show_library
    settings.plex_user_id = plex_user_id or None
    LOGGER.debug(
        "Persisted Plex settings",
        extra={
            "base_url": base_url,
            "movie_library": movie_library,
            "show_library": show_library,
            "plex_user_id": plex_user_id,
            "env_path": str(ENV_PATH),
        },
    )


def _validate_library_choice(library_name: str, available: Dict[str, List[str]], media_type: str) -> str:
    cleaned = (library_name or "").strip()
    if cleaned and cleaned in available.get(media_type, []):
        return cleaned
    defaults = {"movie": "Movies", "show": "TV Shows"}
    return available.get(media_type, [defaults[media_type]])[0]


def list_available_libraries() -> Dict[str, List[str]]:
    if not settings.is_plex_configured:
        LOGGER.info("Plex configuration missing; cannot list libraries")
        return {"movie": [], "show": []}
    try:
        server = PlexServer(str(settings.plex_base_url), settings.plex_token)
        LOGGER.debug(
            "Connected to Plex to list libraries",
            extra={"base_url": str(settings.plex_base_url)},
        )
        return _collect_library_names(server)
    except Exception as exc:
        LOGGER.exception("Unable to load libraries from Plex", extra={"error": str(exc)})
        raise RuntimeError("Unable to load libraries from Plex") from exc


def save_library_preferences(movie_library: str, show_library: str, plex_user_id: str | None = None):
    if not settings.is_plex_configured:
        raise RuntimeError("Plex must be configured before updating libraries")

    available = list_available_libraries()
    movie = _validate_library_choice(movie_library, available, media_type="movie")
    show = _validate_library_choice(show_library, available, media_type="show")
    user_id = (plex_user_id or "").strip() or None

    _persist_settings(str(settings.plex_base_url), str(settings.plex_token), movie, show, user_id)
    LOGGER.info(
        "Updated Plex library preferences",
        extra={"movie_library": movie, "show_library": show, "plex_user_id": user_id},
    )
    return {"movie": movie, "show": show, "plex_user_id": user_id}


def check_login(pin_id: str) -> PlexLoginStatus:
    completed = _COMPLETED_LOGINS.get(pin_id)
    if completed:
        status, cached_at = completed
        if datetime.utcnow() - cached_at <= _COMPLETED_RETENTION:
            LOGGER.debug(
                "Login status replayed from cache",
                extra={"pin_id": pin_id, "status": status.status},
            )
            return status
        _COMPLETED_LOGINS.pop(pin_id, None)

    entry = _PENDING_LOGINS.get(pin_id)
    if not entry:
        LOGGER.debug("Login status check for unknown pinId", extra={"pin_id": pin_id})
        return PlexLoginStatus("invalid")

    pin_login, created_at = entry
    if datetime.utcnow() - created_at > _LOGIN_TIMEOUT:
        LOGGER.info("Plex login attempt expired", extra={"pin_id": pin_id})
        _remove_expired(pin_id)
        return PlexLoginStatus("expired")

    try:
        pin_login.checkLogin()
    except Exception:
        LOGGER.exception("Error during Plex login check", extra={"pin_id": pin_id})
        _remove_expired(pin_id)
        return PlexLoginStatus("error")

    if pin_login.expired:
        LOGGER.info("Plex login pin expired", extra={"pin_id": pin_id})
        _remove_expired(pin_id)
        return PlexLoginStatus("expired")

    if not pin_login.token:
        LOGGER.debug("Plex login still pending", extra={"pin_id": pin_id})
        return PlexLoginStatus("pending")

    account = MyPlexAccount(token=pin_login.token)
    LOGGER.debug("Created Plex account from pin login", extra={"pin_id": pin_id})
    server = _connect_server(account)
    if not server:
        LOGGER.error("Unable to connect to Plex server after login", extra={"pin_id": pin_id})
        _remove_expired(pin_id)
        return PlexLoginStatus("error")

    libraries = _collect_library_names(server)
    movie_library = _validate_library_choice(settings.plex_movie_library, libraries, media_type="movie")
    show_library = _validate_library_choice(settings.plex_show_library, libraries, media_type="show")
    base_url = server.url("") if callable(getattr(server, "url", None)) else str(server.url)
    _persist_settings(
        base_url,
        pin_login.token,
        movie_library,
        show_library,
        settings.plex_user_id,
    )
    LOGGER.info(
        "Plex login authorized",
        extra={
            "pin_id": pin_id,
            "server_name": getattr(server, "friendlyName", None),
            "base_url": base_url,
            "movie_library": movie_library,
            "show_library": show_library,
        },
    )
    status = PlexLoginStatus(
        "authorized",
        token=pin_login.token,
        base_url=base_url,
        server_name=getattr(server, "friendlyName", None),
        libraries=libraries,
    )
    _store_completed(pin_id, status)
    _remove_expired(pin_id)
    return status
