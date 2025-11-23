from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from app.services.plex_service import PlexService

LOG_PATH = Path("/config/tautulli.log")
TV_CONFIG_PATH = Path("/config/tv.yml")
_DEF_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def _normalize_title(value: str | None) -> str:
    return (value or "").strip().lower()


def _serialize_datetime(value: datetime | None) -> str | None:
    try:
        return value.isoformat()
    except Exception:  # noqa: BLE001
        return None


def _serialize_show_entry(item, *, watched_at: datetime | None = None, added_at: datetime | None = None):
    return {
        "title": getattr(item, "title", None),
        "rating_key": getattr(item, "ratingKey", None),
        "library": getattr(item, "librarySectionTitle", None),
        "watched_at": _serialize_datetime(watched_at),
        "added_at": _serialize_datetime(added_at),
    }


def _load_tv_allowlist(path: Path = TV_CONFIG_PATH) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = yaml.safe_load(path.read_text()) or []
    except Exception:  # noqa: BLE001
        return set()

    titles: set[str] = set()

    def _collect(node):
        if isinstance(node, str):
            value = _normalize_title(node)
            if value:
                titles.add(value)
        elif isinstance(node, dict):
            for value in node.values():
                _collect(value)
        elif isinstance(node, Iterable):
            for value in node:
                _collect(value)

    _collect(data)
    return titles


def get_tautulli_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.touch(exist_ok=True)

    logger = logging.getLogger("tautulli")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(_DEF_FORMAT)
        handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def _filter_for_allowlist(
    entries: Sequence[tuple[object, datetime]],
    added: Sequence[tuple[object, datetime]],
    allowlist: set[str],
):
    if not allowlist:
        return []

    filtered = []
    seen = set()
    for item, timestamp in entries:
        rating_key = getattr(item, "ratingKey", None)
        if rating_key is None or rating_key in seen:
            continue
        seen.add(rating_key)
        if _normalize_title(getattr(item, "title", None)) not in allowlist:
            continue
        filtered.append(_serialize_show_entry(item, watched_at=timestamp))
    for item, timestamp in added:
        rating_key = getattr(item, "ratingKey", None)
        if rating_key is None or rating_key in seen:
            continue
        seen.add(rating_key)
        if _normalize_title(getattr(item, "title", None)) not in allowlist:
            continue
        filtered.append(_serialize_show_entry(item, added_at=timestamp))
    return filtered


def log_recent_tv_activity(
    plex: PlexService,
    *,
    user_id: str | None,
    days: int = 7,
) -> None:
    logger = get_tautulli_logger()

    recent_watched = plex.tautulli_recent_show_entries(days=days, max_results=200)
    if recent_watched is None:
        logger.info("Skipping Tautulli TV activity logging; Tautulli not configured")
        return

    recent_added = plex.recently_added_shows(days=days, max_results=200)
    allowlist = _load_tv_allowlist()

    watched_payload = [_serialize_show_entry(item, watched_at=dt) for item, dt in recent_watched]
    added_payload = [_serialize_show_entry(item, added_at=dt) for item, dt in recent_added]

    filtered_payload = _filter_for_allowlist(recent_watched, recent_added, allowlist)

    logger.info(
        "Collected recent Tautulli show activity",
        extra={
            "user_id": user_id or "all",
            "days": days,
            "watched_count": len(watched_payload),
            "added_count": len(added_payload),
            "watched_shows": watched_payload,
            "recently_added_shows": added_payload,
            "tv_yaml_path": str(TV_CONFIG_PATH),
            "tv_allowlist_size": len(allowlist),
        },
    )
    logger.info(
        "Filtered Tautulli show list for app consumption",
        extra={
            "user_id": user_id or "all",
            "days": days,
            "filtered_count": len(filtered_payload),
            "tv_yaml_path": str(TV_CONFIG_PATH),
            "tv_allowlist_size": len(allowlist),
            "filtered_shows": filtered_payload,
        },
    )
