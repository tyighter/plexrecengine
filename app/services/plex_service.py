from datetime import datetime, timedelta
import inspect
import random
from typing import Iterable, List, Optional

from plexapi.server import PlexServer

from app.config import settings
from app.services.plex_logging import get_plex_logger


LOGGER = get_plex_logger()


class PlexService:
    def __init__(self) -> None:
        if not settings.is_plex_configured:
            raise RuntimeError("Plex is not configured. Please sign in through the web interface.")
        self.client = PlexServer(str(settings.plex_base_url), settings.plex_token)
        self._history_params = set(inspect.signature(self.client.history).parameters)
        LOGGER.debug(
            "Initialized Plex service client",
            extra={
                "base_url": str(settings.plex_base_url),
                "movie_library": settings.plex_movie_library,
                "show_library": settings.plex_show_library,
            },
        )

    def _history_kwargs(self, section, cutoff: datetime, limit: int):
        kwargs = {
            "librarySectionID": getattr(section, "key", None),
            "mindate": cutoff,
            "maxresults": limit,
        }

        if "type" in self._history_params:
            section_type = getattr(section, "TYPE", None)
            if section_type:
                kwargs["type"] = section_type

        return kwargs

    def _library_sections(self):
        for name in [settings.plex_movie_library, settings.plex_show_library]:
            try:
                section = self.client.library.section(name)
                LOGGER.debug("Loaded Plex library section", extra={"section": name})
                yield section
            except Exception:
                LOGGER.exception("Failed to load Plex library section", extra={"section": name})
                continue

    def recently_watched_movies(self, days: int = 30, max_results: Optional[int] = None):
        cutoff = datetime.now() - timedelta(days=days)
        limit = max_results or 200
        movies = []
        for section in self._library_sections():
            if section.TYPE != "movie":
                continue

            try:
                history_entries = self.client.history(**self._history_kwargs(section, cutoff, limit))
            except Exception:
                LOGGER.exception(
                    "Failed to load Plex history for movies", extra={"section": getattr(section, "title", None)}
                )
                history_entries = []

            if not history_entries:
                try:
                    history_entries = section.search(
                        sort="lastViewedAt:desc", unwatched=False, maxresults=limit
                    )
                except Exception:
                    LOGGER.exception(
                        "Failed Plex search fallback for movies",
                        extra={"section": getattr(section, "title", None)},
                    )
                    history_entries = []

            for entry in history_entries:
                movie = getattr(entry, "item", entry)
                entry_type = getattr(movie, "type", None) or getattr(entry, "type", None)
                if entry_type and entry_type != "movie":
                    continue

                last_viewed = getattr(movie, "lastViewedAt", None) or getattr(entry, "viewedAt", None)
                if last_viewed and last_viewed >= cutoff:
                    movies.append(movie)

        LOGGER.debug(
            "Collected recently watched movies",
            extra={"count": len(movies), "cutoff": cutoff.isoformat()},
        )
        sorted_items = sorted(
            movies,
            key=lambda m: getattr(m, "lastViewedAt", None) or getattr(m, "viewedAt", None) or datetime.min,
            reverse=True,
        )
        if max_results:
            return sorted_items[:max_results]
        return sorted_items

    def recently_watched_shows(self, days: int = 30, max_results: Optional[int] = None):
        cutoff = datetime.now() - timedelta(days=days)
        limit = max_results or 200
        episodes = []
        for section in self._library_sections():
            if section.TYPE != "show":
                continue

            try:
                history_entries = self.client.history(**self._history_kwargs(section, cutoff, limit))
            except Exception:
                LOGGER.exception(
                    "Failed to load Plex history for shows", extra={"section": getattr(section, "title", None)}
                )
                history_entries = []

            if not history_entries:
                try:
                    history_entries = section.search(
                        sort="lastViewedAt:desc", unwatched=False, maxresults=limit
                    )
                except Exception:
                    LOGGER.exception(
                        "Failed Plex search fallback for shows",
                        extra={"section": getattr(section, "title", None)},
                    )
                    history_entries = []

            for entry in history_entries:
                episode = getattr(entry, "item", entry)
                entry_type = getattr(episode, "type", None) or getattr(entry, "type", None)
                if entry_type and entry_type != "episode":
                    continue

                last_viewed = getattr(episode, "lastViewedAt", None) or getattr(entry, "viewedAt", None)
                if last_viewed and last_viewed >= cutoff:
                    episodes.append(episode)

        LOGGER.debug(
            "Collected recently watched episodes",
            extra={"count": len(episodes), "cutoff": cutoff.isoformat()},
        )
        sorted_eps = sorted(
            episodes,
            key=lambda e: getattr(e, "lastViewedAt", None) or getattr(e, "viewedAt", None) or datetime.min,
            reverse=True,
        )
        shows = []
        seen_keys = set()
        for ep in sorted_eps:
            show_title = getattr(ep, "grandparentTitle", None) or getattr(ep, "title", None)
            if not show_title or show_title in seen_keys:
                continue
            seen_keys.add(show_title)

            if getattr(ep, "type", None) == "show":
                show_item = ep
            else:
                try:
                    show_item = ep.show()
                except Exception:
                    continue

            shows.append(show_item)
            if max_results and len(shows) >= max_results:
                break
        LOGGER.debug(
            "Selected unique recently watched shows", extra={"count": len(shows)}
        )
        return shows

    def search_unwatched(self, section_type: str, query: str):
        for section in self._library_sections():
            if section.TYPE != section_type:
                continue
            try:
                # PlexAPI does not expose a searchTitle helper; use the standard
                # search API with explicit title and unwatched filters instead.
                for item in section.search(title=query, unwatched=True):
                    yield item
            except Exception:
                LOGGER.exception(
                    "Failed Plex search", extra={"section_type": section_type, "query": query}
                )
                continue

    def _find_section_for_item(self, item):
        section_id = getattr(item, "librarySectionID", None)
        if section_id is not None:
            try:
                return self.client.library.sectionByID(section_id)
            except Exception:
                LOGGER.exception(
                    "Unable to load section by ID for Plex item",
                    extra={"section_id": section_id},
                )
                pass

        item_type = getattr(item, "type", None)
        if item_type:
            for section in self._library_sections():
                if section.TYPE == item_type:
                    return section
        return None

    def ensure_collection(self, title: str, section, items=None):
        existing = next((c for c in section.collections() if c.title == title), None)
        LOGGER.debug(
            "Ensuring Plex collection exists", extra={"title": title, "section": getattr(section, "title", None)}
        )
        if existing:
            return existing

        return section.createCollection(title, items=items)

    def set_collection_members(self, collection_name: str, items: Iterable):
        items = list(items)
        if not items:
            return

        section = self._find_section_for_item(items[0])
        if section is None:
            raise RuntimeError("Unable to determine library section for collection members")

        collection = self.ensure_collection(collection_name, section, items=items)
        try:
            existing_items = list(collection.items())
        except Exception:
            LOGGER.exception(
                "Failed to load existing Plex collection items",
                extra={"collection": collection_name},
            )
            existing_items = []

        target_keys = {
            getattr(item, "ratingKey", None) for item in items if hasattr(item, "ratingKey")
        }
        existing_keys = {
            getattr(item, "ratingKey", None) for item in existing_items if hasattr(item, "ratingKey")
        }

        items_to_remove = []
        if target_keys:
            for item in existing_items:
                key = getattr(item, "ratingKey", None)
                if key is not None and key not in target_keys:
                    items_to_remove.append(item)

        if items_to_remove:
            try:
                collection.removeItems(items_to_remove)
                LOGGER.debug(
                    "Removed items missing from recommendations",
                    extra={
                        "collection": collection_name,
                        "removed_count": len(items_to_remove),
                        "remaining_count": len(existing_items) - len(items_to_remove),
                    },
                )
            except Exception:
                LOGGER.exception(
                    "Failed to remove outdated Plex collection items",
                    extra={"collection": collection_name, "remove_count": len(items_to_remove)},
                )

        new_items = []
        for item in items:
            key = getattr(item, "ratingKey", None)
            if key is not None and key in existing_keys:
                continue
            new_items.append(item)
            if key is not None:
                existing_keys.add(key)

        if not new_items:
            LOGGER.debug(
                "No new items to add to Plex collection",
                extra={
                    "collection": collection_name,
                    "existing_count": len(existing_items),
                    "removed_count": len(items_to_remove),
                },
            )
            return

        random.shuffle(new_items)

        for item in new_items:
            item.addCollection(collection)
            LOGGER.debug(
                "Added item to Plex collection",
                extra={
                    "collection": collection_name,
                    "item_title": getattr(item, "title", None),
                    "section": getattr(section, "title", None),
                },
            )

        LOGGER.debug(
            "Updated Plex collection members",
            extra={
                "collection": collection_name,
                "added_count": len(new_items),
                "removed_count": len(items_to_remove),
                "existing_count": len(existing_items),
            },
        )

    def poster_url(self, item) -> Optional[str]:
        """Return a usable poster URL for any Plex item.

        Plex items may expose their artwork in different attributes depending on
        whether they are movies, shows, episodes, or collections. We try a few
        options so the UI can display a poster even when the primary thumb is
        missing. If the attribute already contains a fully-qualified URL, return
        it directly; otherwise build the server URL so authentication tokens are
        preserved.
        """

        poster_candidates = [
            getattr(item, "thumb", None),
            getattr(item, "parentThumb", None),
            getattr(item, "grandparentThumb", None),
            getattr(item, "art", None),
        ]

        for poster in poster_candidates:
            if not poster:
                continue
            try:
                if isinstance(poster, str) and poster.startswith(("http://", "https://")):
                    return poster
                return self.client.url(poster, includeToken=True)
            except Exception:
                LOGGER.exception(
                    "Failed to resolve Plex poster URL",
                    extra={"poster": poster, "item": getattr(item, "title", None)},
                )
                continue

        return None


def get_plex_service() -> PlexService:
    LOGGER.debug("Creating Plex service instance")
    return PlexService()
