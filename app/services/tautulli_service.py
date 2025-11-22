from __future__ import annotations

import httpx

from app.config import settings
from app.services.plex_logging import get_plex_logger

LOGGER = get_plex_logger()
DEFAULT_TIMEOUT = 10.0


class TautulliClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        if not self.base_url or not self.api_key:
            raise ValueError("Tautulli base URL and API key are required")

    def _request(self, cmd: str, params: dict[str, object] | None = None):
        query: dict[str, object] = {"apikey": self.api_key, "cmd": cmd}
        if params:
            query.update(params)
        url = f"{self.base_url}/api/v2"
        response = httpx.get(url, params=query, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected response from Tautulli")
        response_data = data.get("response") or {}
        if response_data.get("result") != "success":
            message = response_data.get("message") or "Tautulli request failed"
            raise RuntimeError(str(message))
        return response_data.get("data")

    def list_users(self) -> list[dict[str, str]]:
        raw_data = self._request("get_users_table", {"length": 1000})
        entries = []
        if isinstance(raw_data, dict):
            entries = raw_data.get("data", []) or []
        elif isinstance(raw_data, list):
            entries = raw_data

        users: list[dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            user_id = str(entry.get("user_id") or "").strip()
            if not user_id:
                continue
            username = (
                entry.get("friendly_name")
                or entry.get("username")
                or entry.get("email")
                or user_id
            )
            users.append({"id": user_id, "username": str(username)})

        LOGGER.debug("Loaded Tautulli users", extra={"count": len(users)})
        return users


def get_tautulli_client() -> TautulliClient:
    if not settings.is_tautulli_configured:
        raise RuntimeError("Tautulli is not configured")
    return TautulliClient(str(settings.tautulli_base_url), str(settings.tautulli_api_key))


def list_tautulli_users() -> list[dict[str, str]]:
    try:
        client = get_tautulli_client()
        return client.list_users()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unable to load users from Tautulli", extra={"error": str(exc)})
        raise RuntimeError("Unable to load users from Tautulli") from exc
