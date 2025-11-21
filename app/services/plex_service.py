from datetime import datetime, timedelta
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

    def recently_watched_movies(self, days: int = 30, max_results: Optional[int] = None):
        cutoff = datetime.now() - timedelta(days=days)
        movies = []
        for section in self._library_sections():
            if section.TYPE == "movie":
                for movie in section.search(sort="lastViewedAt:desc", unwatched=False):
                    last_viewed = getattr(movie, "lastViewedAt", None)
                    if last_viewed and last_viewed >= cutoff:
                        movies.append(movie)
        sorted_items = sorted(movies, key=lambda m: getattr(m, "lastViewedAt", None) or datetime.min, reverse=True)
        if max_results:
            return sorted_items[:max_results]
        return sorted_items

    def recently_watched_shows(self, days: int = 30, max_results: Optional[int] = None):
        cutoff = datetime.now() - timedelta(days=days)
        episodes = []
        for section in self._library_sections():
            if section.TYPE == "show":
                for episode in section.search(sort="lastViewedAt:desc", unwatched=False):
                    last_viewed = getattr(episode, "lastViewedAt", None)
                    if last_viewed and last_viewed >= cutoff:
                        episodes.append(episode)
        sorted_eps = sorted(episodes, key=lambda e: getattr(e, "lastViewedAt", None) or datetime.min, reverse=True)
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
                continue

        return None


def get_plex_service() -> PlexService:
    return PlexService()
