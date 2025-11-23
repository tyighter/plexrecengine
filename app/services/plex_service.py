from datetime import datetime, timedelta
import random
from typing import Iterable, List, Optional

from plexapi.exceptions import NotFound
from plexapi.server import PlexServer

from app.config import settings
from app.services.plex_logging import get_plex_logger
from app.services.tautulli_service import get_tautulli_client


LOGGER = get_plex_logger()


class PlexService:
    def __init__(self) -> None:
        if not settings.is_plex_configured:
            raise RuntimeError("Plex is not configured. Please sign in through the web interface.")
        self.client = PlexServer(str(settings.plex_base_url), settings.plex_token)
        self.filter_history_by_user = False
        self._plex_account_id: Optional[str] = None

        if settings.plex_user_id:
            if str(settings.plex_user_id).isdigit():
                self.filter_history_by_user = True
                self._plex_account_id = str(settings.plex_user_id)
            else:
                LOGGER.warning(
                    "Ignoring non-numeric Plex user id for history filtering",
                    extra={"plex_user_id": settings.plex_user_id},
                )
        LOGGER.debug(
            "Initialized Plex service client",
            extra={
                "base_url": str(settings.plex_base_url),
                "movie_library": settings.plex_movie_library,
                "show_library": settings.plex_show_library,
                "plex_user_id": settings.plex_user_id,
            },
        )

    def _history_kwargs(
        self, section, cutoff: datetime, limit: int, include_account_filter: bool = True
    ):
        kwargs = {
            "librarySectionID": getattr(section, "key", None),
            "mindate": cutoff,
            "maxresults": limit,
        }

        if include_account_filter and self.filter_history_by_user and self._plex_account_id:
            kwargs["accountID"] = self._plex_account_id
            LOGGER.debug(
                "Filtering Plex history by user",
                extra={
                    "section": getattr(section, "title", None),
                    "account_id": self._plex_account_id,
                    "cutoff": cutoff.isoformat(),
                },
            )

        return kwargs

    def _load_history_entries(self, section, cutoff: datetime, limit: int):
        try:
            return self.client.history(
                **self._history_kwargs(section, cutoff, limit, include_account_filter=True)
            )
        except Exception:
            LOGGER.exception(
                "Failed to load Plex history",
                extra={"section": getattr(section, "title", None)},
            )
            return []

    def _retry_history_without_account(self, section, cutoff: datetime, limit: int):
        if not self.filter_history_by_user:
            return []
        try:
            LOGGER.debug(
                "Retrying Plex history without user filter",
                extra={"section": getattr(section, "title", None)},
            )
            return self.client.history(
                **self._history_kwargs(section, cutoff, limit, include_account_filter=False)
            )
        except Exception:
            LOGGER.exception(
                "Failed to load Plex history without user filter",
                extra={"section": getattr(section, "title", None)},
            )
            return []

    def _recent_search_fallback(self, section, limit: int):
        try:
            return section.search(sort="lastViewedAt:desc", unwatched=False, maxresults=limit)
        except Exception:
            LOGGER.exception(
                "Failed Plex search fallback",
                extra={"section": getattr(section, "title", None)},
            )
            return []

    def _library_sections(self):
        run_id = f"library-refresh-{datetime.now().isoformat()}-{random.randint(1000, 9999)}"
        loaded_sections = []
        for name in [settings.plex_movie_library, settings.plex_show_library]:
            try:
                section = self.client.library.section(name)
                loaded_sections.append(name)
                yield section
            except Exception:
                LOGGER.exception(
                    "Failed to load Plex library section",
                    extra={"section": name, "run_id": run_id},
                )
                continue

        if loaded_sections:
            LOGGER.debug(
                "Loaded Plex library sections",
                extra={"run_id": run_id, "sections": loaded_sections},
            )

    def _tautulli_user_id(self) -> Optional[str]:
        user_id = (settings.tautulli_user_id or settings.plex_user_id or "").strip()
        return user_id or None

    def _load_tautulli_history(
        self, media_type: str, days: int, limit: int, user_id: Optional[str]
    ):
        if not settings.is_tautulli_configured:
            return None
        try:
            client = get_tautulli_client()
        except Exception:
            LOGGER.exception("Failed to initialize Tautulli client")
            return [] if user_id else None

        try:
            cutoff = datetime.now() - timedelta(days=days)
            return client.history(
                media_type=media_type,
                after=int(cutoff.timestamp()),
                limit=limit,
                user_id=user_id,
            )
        except Exception:
            LOGGER.exception(
                "Failed to load history from Tautulli", extra={"media_type": media_type}
            )
            return [] if user_id else None

    def _tautulli_timestamp(self, value) -> datetime:
        try:
            return datetime.fromtimestamp(int(value))
        except Exception:
            return datetime.min

    def recently_watched_movies(self, days: int = 30, max_results: Optional[int] = None):
        cutoff = datetime.now() - timedelta(days=days)
        limit = max_results or 200
        tautulli_user_id = self._tautulli_user_id()
        tautulli_history = self._load_tautulli_history("movie", days, limit, tautulli_user_id)
        if tautulli_history is not None:
            movies = []
            seen_movie_keys = set()
            for entry in tautulli_history:
                rating_key = entry.get("rating_key") or entry.get("parent_rating_key")
                if rating_key is None or rating_key in seen_movie_keys:
                    continue
                movie = self.fetch_item(
                    rating_key,
                    extra={
                        "source": "recently_watched_movies",
                        "context": "tautulli_history",
                        "tautulli_user_id": tautulli_user_id,
                    },
                )
                if not movie or getattr(movie, "type", None) != "movie":
                    continue
                last_viewed = self._tautulli_timestamp(
                    entry.get("last_played")
                    or entry.get("date")
                    or entry.get("stopped")
                )
                if last_viewed < cutoff:
                    continue
                seen_movie_keys.add(rating_key)
                movies.append((movie, last_viewed))
                if max_results and len(movies) >= max_results:
                    break
            movies.sort(key=lambda pair: pair[1], reverse=True)
            return [item for item, _ in movies][: max_results or len(movies)]

        movies = []
        seen_movie_keys = set()
        for section in self._library_sections():
            if section.TYPE != "movie":
                continue

            history_entries = self._load_history_entries(section, cutoff, limit)

            if not history_entries:
                history_entries = self._retry_history_without_account(section, cutoff, limit)

            if not history_entries:
                history_entries = self._recent_search_fallback(section, limit)

            for entry in history_entries:
                movie = getattr(entry, "item", entry)
                entry_type = getattr(movie, "type", None) or getattr(entry, "type", None)
                if entry_type and entry_type != "movie":
                    continue

                last_viewed = getattr(movie, "lastViewedAt", None) or getattr(entry, "viewedAt", None)
                rating_key = getattr(movie, "ratingKey", None)
                if last_viewed and last_viewed >= cutoff:
                    if rating_key is not None and rating_key in seen_movie_keys:
                        continue
                    if rating_key is not None:
                        seen_movie_keys.add(rating_key)
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
        tautulli_user_id = self._tautulli_user_id()
        tautulli_history = self._load_tautulli_history("episode", days, limit, tautulli_user_id)
        if tautulli_history is not None:
            shows_with_dates = []
            seen_show_keys = set()
            for entry in tautulli_history:
                rating_key = (
                    entry.get("grandparent_rating_key")
                    or entry.get("rating_key")
                    or entry.get("parent_rating_key")
                )
                if rating_key is None or rating_key in seen_show_keys:
                    continue
                watched_at = self._tautulli_timestamp(
                    entry.get("last_played")
                    or entry.get("date")
                    or entry.get("stopped")
                )
                if watched_at < cutoff:
                    continue
                item = self.fetch_item(
                    rating_key,
                    extra={
                        "source": "recently_watched_shows",
                        "context": "tautulli_history",
                        "tautulli_user_id": tautulli_user_id,
                    },
                )
                if item is None:
                    continue
                if getattr(item, "type", None) == "episode":
                    try:
                        item = item.show()
                    except Exception:
                        continue
                if getattr(item, "type", None) != "show":
                    continue
                seen_show_keys.add(rating_key)
                shows_with_dates.append((item, watched_at))
                if max_results and len(shows_with_dates) >= max_results:
                    break
            shows_with_dates.sort(key=lambda pair: pair[1], reverse=True)
            return [item for item, _ in shows_with_dates][: max_results or len(shows_with_dates)]

        episodes = []
        seen_episode_keys = set()
        for section in self._library_sections():
            if section.TYPE != "show":
                continue

            history_entries = self._load_history_entries(section, cutoff, limit)

            if not history_entries:
                history_entries = self._retry_history_without_account(section, cutoff, limit)

            if not history_entries:
                history_entries = self._recent_search_fallback(section, limit)

            for entry in history_entries:
                episode = getattr(entry, "item", entry)
                entry_type = getattr(episode, "type", None) or getattr(entry, "type", None)
                if entry_type and entry_type != "episode":
                    continue

                last_viewed = getattr(episode, "lastViewedAt", None) or getattr(entry, "viewedAt", None)
                rating_key = getattr(episode, "ratingKey", None)
                if last_viewed and last_viewed >= cutoff:
                    if rating_key is not None and rating_key in seen_episode_keys:
                        continue
                    if rating_key is not None:
                        seen_episode_keys.add(rating_key)
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

    def iter_library_items(self, section_type: str):
        for section in self._library_sections():
            if section.TYPE != section_type:
                continue
            try:
                for item in section.all():
                    yield item
            except Exception:
                LOGGER.exception(
                    "Failed to iterate Plex library items", extra={"section": getattr(section, "title", None)}
                )

    def recently_added(self, media_type: str, max_results: int = 50):
        for section in self._library_sections():
            if section.TYPE != media_type:
                continue
            try:
                yield from section.search(sort="addedAt:desc", maxresults=max_results)
            except Exception:
                LOGGER.exception(
                    "Failed to load recently added items", extra={"section": getattr(section, "title", None)}
                )

    def search_library(self, section_type: str, query: str):
        for section in self._library_sections():
            if section.TYPE != section_type:
                continue
            try:
                # PlexAPI does not expose a searchTitle helper; use the standard
                # search API with explicit title filter instead so watched items
                # are included in recommendations.
                for item in section.search(title=query, maxresults=50):
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

    def fetch_item(self, rating_key, *, extra: Optional[dict] = None):
        """Fetch a Plex item by rating key or metadata path.

        Some Plex API calls expect the full metadata path (``/library/metadata/<id>``)
        while others accept a bare rating key. Normalize the input so collection
        updates can reliably fetch every recommendation, and log failures so we
        can surface partial updates instead of silently skipping entries. The
        ``extra`` mapping, when provided, is merged into the log context so
        callers can tag the source of a missing item.
        """

        key = str(rating_key)
        if key.isdigit():
            key = f"/library/metadata/{key}"

        extra_data = {"rating_key": rating_key, "resolved_key": key}
        if extra:
            extra_data.update(extra)

        try:
            return self.client.fetchItem(key)
        except NotFound:
            context_bits = []
            for field, value in extra_data.items():
                if value is None:
                    continue
                context_bits.append(f"{field}={value}")

            context_suffix = f" ({', '.join(context_bits)})" if context_bits else ""
            LOGGER.error(
                "Plex item not found%s", context_suffix, extra=extra_data
            )
            return None
        except Exception:
            LOGGER.exception(
                "Failed to fetch Plex item by rating key",
                extra=extra_data,
            )
            return None

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

        try:
            collection.addItems(new_items)
        except Exception:
            LOGGER.exception(
                "Failed to add items to Plex collection",
                extra={
                    "collection": collection_name,
                    "add_count": len(new_items),
                    "section": getattr(section, "title", None),
                },
            )
            return

        LOGGER.debug(
            "Added items to Plex collection",
            extra={
                "collection": collection_name,
                "added_count": len(new_items),
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

    def related_items(self, item, media_type: str, limit: Optional[int] = None):
        """Return Plex-provided related items for a given media item."""

        if item is None:
            return []

        try:
            related_fn = getattr(item, "related", None)
            if callable(related_fn):
                related_items = list(related_fn())
            else:
                related_items = self._fetch_related_via_metadata(item)
        except Exception:
            LOGGER.exception(
                "Failed to fetch Plex related items", extra={"title": getattr(item, "title", None)}
            )
            return []

        filtered: list = []
        for candidate in related_items:
            if media_type and getattr(candidate, "type", None) != media_type:
                continue
            filtered.append(candidate)
            if limit and len(filtered) >= limit:
                break

        LOGGER.debug(
            "Loaded Plex related items", extra={"title": getattr(item, "title", None), "count": len(filtered)}
        )
        return filtered

    def _fetch_related_via_metadata(self, item):
        title = getattr(item, "title", None)
        key = getattr(item, "key", None)

        if not key:
            LOGGER.warning(
                "Plex item is missing key for related lookup",
                extra={"title": title},
            )
            return []

        related_items: list = []
        try:
            related_items = list(self.client.fetchItems(f"{key}/related"))
            if related_items:
                LOGGER.debug(
                    "Fetched related items via fallback endpoint",
                    extra={"title": title, "count": len(related_items)},
                )
                return related_items
        except Exception:
            LOGGER.exception(
                "Failed to fetch Plex related items via fallback",
                extra={"title": title},
            )

        try:
            metadata_xml = self.client.query(key)
        except Exception:
            LOGGER.exception(
                "Failed to query Plex metadata for related items",
                extra={"title": title},
            )
            return []

        related_paths = []
        for element in metadata_xml.iter():
            element_key = element.attrib.get("key") if hasattr(element, "attrib") else None
            if not element_key:
                continue
            if "/related" not in element_key:
                continue
            if element_key not in related_paths:
                related_paths.append(element_key)

        if not related_paths:
            LOGGER.debug(
                "No related paths discovered in metadata XML",
                extra={"title": title},
            )
            return []

        for related_path in related_paths:
            path = related_path
            if not path.startswith("/"):
                path = f"{key}/{path}".replace("//", "/", 1)
            try:
                related_items = list(self.client.fetchItems(path))
            except Exception:
                LOGGER.exception(
                    "Failed to fetch Plex related items from metadata path",
                    extra={"title": title, "path": path},
                )
                continue

            if related_items:
                LOGGER.debug(
                    "Fetched related items from metadata XML path",
                    extra={"title": title, "path": path, "count": len(related_items)},
                )
                return related_items

        LOGGER.debug(
            "No related items returned from any metadata paths",
            extra={"title": title, "paths": related_paths},
        )
        return []

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
