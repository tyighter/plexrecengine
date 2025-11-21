from datetime import datetime
from typing import Iterable, List, Optional

from plexapi.server import PlexServer

from app.config import settings


class PlexService:
    def __init__(self) -> None:
        if not settings.is_plex_configured:
            raise RuntimeError("Plex is not configured. Please sign in through the web interface.")
        self.client = PlexServer(str(settings.plex_base_url), settings.plex_token)

    def _library_sections(self):
        for name in [settings.plex_movie_library, settings.plex_show_library]:
            try:
                yield self.client.library.section(name)
            except Exception:
                continue

    def recently_watched_movies(self, limit: int = 10):
        movies = []
        for section in self._library_sections():
            if section.TYPE == "movie":
                movies.extend(section.search(sort="lastViewedAt:desc", unwatched=False)[: limit * 2])
        sorted_items = sorted(movies, key=lambda m: getattr(m, "lastViewedAt", None) or datetime.min, reverse=True)
        return sorted_items[:limit]

    def recently_watched_shows(self, limit: int = 10):
        episodes = []
        for section in self._library_sections():
            if section.TYPE == "show":
                episodes.extend(section.search(sort="lastViewedAt:desc", unwatched=False)[: limit * 3])
        sorted_eps = sorted(episodes, key=lambda e: getattr(e, "lastViewedAt", None) or datetime.min, reverse=True)
        shows = []
        seen_keys = set()
        for ep in sorted_eps:
            show = ep.grandparentTitle
            if show in seen_keys:
                continue
            seen_keys.add(show)
            shows.append(ep.show())
            if len(shows) >= limit:
                break
        return shows

    def search_unwatched(self, section_type: str, query: str):
        for section in self._library_sections():
            if section.TYPE != section_type:
                continue
            try:
                for item in section.searchTitle(query, unwatched=True):
                    yield item
            except Exception:
                continue

    def _find_section_for_item(self, item):
        section_id = getattr(item, "librarySectionID", None)
        if section_id is not None:
            try:
                return self.client.library.sectionByID(section_id)
            except Exception:
                pass

        item_type = getattr(item, "type", None)
        if item_type:
            for section in self._library_sections():
                if section.TYPE == item_type:
                    return section
        return None

    def ensure_collection(self, title: str, section):
        existing = next((c for c in section.collections() if c.title == title), None)
        return existing or section.createCollection(title)

    def set_collection_members(self, collection_name: str, items: Iterable):
        items = list(items)
        if not items:
            return

        section = self._find_section_for_item(items[0])
        if section is None:
            raise RuntimeError("Unable to determine library section for collection members")

        collection = self.ensure_collection(collection_name, section)
        collection.deleteItems(collection.items())
        for item in items:
            item.addCollection(collection)

    def poster_url(self, item) -> Optional[str]:
        try:
            return self.client.url(item.thumb) if item.thumb else None
        except Exception:
            return None


def get_plex_service() -> PlexService:
    return PlexService()
