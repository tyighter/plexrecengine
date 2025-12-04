from datetime import datetime, timedelta
import inspect
import random
from typing import Iterable, List, Optional

from plexapi.exceptions import NotFound
from plexapi.server import PlexServer

from app.config import settings
from app.services.generate_logging import get_collections_logger
from app.services.plex_logging import get_plex_logger
from app.services.tautulli_service import get_tautulli_client


LOGGER = get_plex_logger()
COLLECTION_LOGGER = get_collections_logger()


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
        tautulli_shows = self._recent_shows_from_tautulli(days, limit)
        if tautulli_shows is not None:
            trimmed = tautulli_shows[: max_results or len(tautulli_shows)]
            return [item for item, _ in trimmed]

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

    def _recent_shows_from_tautulli(self, days: int, limit: int):
        cutoff = datetime.now() - timedelta(days=days)
        tautulli_user_id = self._tautulli_user_id()
        tautulli_history = self._load_tautulli_history("episode", days, limit, tautulli_user_id)
        if tautulli_history is None:
            return None

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
                entry.get("last_played") or entry.get("date") or entry.get("stopped")
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
            if limit and len(shows_with_dates) >= limit:
                break

        shows_with_dates.sort(key=lambda pair: pair[1], reverse=True)
        return shows_with_dates

    def tautulli_recent_show_entries(self, days: int = 7, max_results: Optional[int] = None):
        limit = max_results or 200
        return self._recent_shows_from_tautulli(days, limit)

    def recently_added_shows(self, days: int = 7, max_results: int = 200):
        cutoff = datetime.now() - timedelta(days=days)
        shows_with_dates = []
        seen_rating_keys = set()
        for item in self.recently_added("show", max_results=max_results):
            added_at = getattr(item, "addedAt", None) or datetime.min
            if added_at < cutoff:
                continue
            show_item = item
            if getattr(item, "type", None) != "show":
                try:
                    show_item = item.show()
                except Exception:
                    continue
            rating_key = getattr(show_item, "ratingKey", None)
            if rating_key is None or rating_key in seen_rating_keys:
                continue
            seen_rating_keys.add(rating_key)
            shows_with_dates.append((show_item, added_at))
        shows_with_dates.sort(key=lambda pair: pair[1], reverse=True)
        return shows_with_dates

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
        if existing:
            LOGGER.debug(
                "Ensuring Plex collection exists",
                extra={"title": title, "section": getattr(section, "title", None)},
            )
            COLLECTION_LOGGER.debug(
                "Ensuring Plex collection exists",
                extra={"title": title, "section": getattr(section, "title", None)},
            )
            return self._ensure_collection_custom_capabilities(existing, title, section)

        LOGGER.info(
            "Creating new Plex collection",
            extra={"title": title, "section": getattr(section, "title", None)},
        )
        COLLECTION_LOGGER.info(
            "Creating new Plex collection",
            extra={"title": title, "section": getattr(section, "title", None)},
        )
        created = section.createCollection(title, items=items)
        return self._ensure_collection_custom_capabilities(created, title, section)

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
        COLLECTION_LOGGER.info(
            "Updating collection members",
            extra={"collection": collection_name, "item_count": len(items)},
        )
        if not items:
            COLLECTION_LOGGER.warning(
                "Skipping empty collection update", extra={"collection": collection_name}
            )
            return

        section = self._find_section_for_item(items[0])
        if section is None:
            raise RuntimeError("Unable to determine library section for collection members")

        collection = self.ensure_collection(collection_name, section, items=items)
        self._set_collection_custom_sort(collection, collection_name)

        ordered_metadata = [
            {
                "position": index,
                "title": getattr(item, "title", None),
                "rating_key": getattr(item, "ratingKey", None),
            }
            for index, item in enumerate(items, start=1)
        ]
        COLLECTION_LOGGER.info(
            "Prepared ordered Plex collection items",
            extra={
                "collection": collection_name,
                "item_count": len(ordered_metadata),
                "items": ordered_metadata,
            },
        )

        try:
            existing_items = list(collection.items())
            COLLECTION_LOGGER.debug(
                "Loaded existing Plex collection items",
                extra={"collection": collection_name, "existing_count": len(existing_items)},
            )
        except Exception:
            LOGGER.exception(
                "Failed to load existing Plex collection items",
                extra={"collection": collection_name},
            )
            COLLECTION_LOGGER.exception(
                "Failed to load existing Plex collection items",
                extra={"collection": collection_name},
            )
            existing_items = []

        desired_keys = [
            getattr(item, "ratingKey", None) for item in items if hasattr(item, "ratingKey")
        ]
        existing_keys = [
            getattr(item, "ratingKey", None)
            for item in existing_items
            if hasattr(item, "ratingKey")
        ]

        if desired_keys == existing_keys and len(desired_keys) == len(items):
            LOGGER.debug(
                "Plex collection already up to date",
                extra={
                    "collection": collection_name,
                    "item_count": len(items),
                },
            )
            COLLECTION_LOGGER.info(
                "Plex collection already up to date",
                extra={"collection": collection_name, "item_count": len(items)},
            )
            return

        supports_reorder = callable(getattr(collection, "reorderItems", None))

        try:
            if existing_items:
                collection.removeItems(existing_items)
                LOGGER.debug(
                    "Cleared existing Plex collection items",
                    extra={
                        "collection": collection_name,
                        "removed_count": len(existing_items),
                    },
                )
                COLLECTION_LOGGER.debug(
                    "Cleared existing Plex collection items",
                    extra={
                        "collection": collection_name,
                        "removed_count": len(existing_items),
                    },
                )

            if supports_reorder:
                collection.addItems(items)
            else:
                for index, item in enumerate(items, start=1):
                    collection.addItems([item])
                    COLLECTION_LOGGER.info(
                        "Added Plex collection item sequentially",
                        extra={
                            "collection": collection_name,
                            "position": index,
                            "title": getattr(item, "title", None),
                            "rating_key": getattr(item, "ratingKey", None),
                        },
                    )
        except Exception:
            LOGGER.exception(
                "Failed to replace Plex collection items",
                extra={
                    "collection": collection_name,
                    "target_count": len(items),
                    "section": getattr(section, "title", None),
                },
            )
            COLLECTION_LOGGER.exception(
                "Failed to replace Plex collection items",
                extra={
                    "collection": collection_name,
                    "target_count": len(items),
                    "section": getattr(section, "title", None),
                },
            )
            return

        reordered = (
            self.reorder_collection_items(collection, items, collection_name)
            if supports_reorder
            else True
        )
        if not reordered:
            LOGGER.warning(
                "Unable to apply custom order to Plex collection; items may not match UI ordering",
                extra={"collection": collection_name, "item_count": len(items)},
            )
            COLLECTION_LOGGER.warning(
                "Unable to apply custom order to Plex collection; items may not match UI ordering",
                extra={"collection": collection_name, "item_count": len(items)},
            )
        elif not supports_reorder:
            COLLECTION_LOGGER.info(
                "Plex collection does not support reordering; added items sequentially to preserve configured order",
                extra={"collection": collection_name, "item_count": len(items)},
            )

        LOGGER.debug(
            "Rebuilt Plex collection items in configured order",
            extra={
                "collection": collection_name,
                "item_count": len(items),
                "section": getattr(section, "title", None),
            },
        )
        COLLECTION_LOGGER.info(
            "Rebuilt Plex collection items in configured order",
            extra={
                "collection": collection_name,
                "item_count": len(items),
                "section": getattr(section, "title", None),
            },
        )

    def _set_collection_custom_sort(self, collection, collection_name: str) -> bool:
        sort_kwargs = {"sort": "custom", "collectionSort": "custom"}
        sort_update = getattr(collection, "sortUpdate", None)
        if callable(sort_update):
            filtered_kwargs = self._filter_callable_kwargs(sort_update, sort_kwargs)
            try:
                sort_update(**filtered_kwargs)
                collection.reload()
                LOGGER.debug(
                    "Configured Plex collection sort order to custom",
                    extra={"collection": collection_name},
                )
                COLLECTION_LOGGER.debug(
                    "Configured Plex collection sort order to custom",
                    extra={"collection": collection_name},
                )
                return True
            except Exception:
                LOGGER.exception(
                    "Failed to configure Plex collection sort order via sortUpdate",
                    extra={"collection": collection_name},
                )
                COLLECTION_LOGGER.exception(
                    "Failed to configure Plex collection sort order via sortUpdate",
                    extra={"collection": collection_name},
                )

        edit_fn = getattr(collection, "edit", None)
        if callable(edit_fn):
            filtered_kwargs = self._filter_callable_kwargs(edit_fn, sort_kwargs)
            try:
                edit_fn(**filtered_kwargs)
                collection.reload()
                LOGGER.debug(
                    "Configured Plex collection sort order to custom via edit",
                    extra={"collection": collection_name},
                )
                COLLECTION_LOGGER.debug(
                    "Configured Plex collection sort order to custom via edit",
                    extra={"collection": collection_name},
                )
                return True
            except Exception:
                LOGGER.exception(
                    "Failed to configure Plex collection sort order via edit",
                    extra={"collection": collection_name},
                )
                COLLECTION_LOGGER.exception(
                    "Failed to configure Plex collection sort order via edit",
                    extra={"collection": collection_name},
                )

        LOGGER.warning(
            "Unable to set Plex collection sort order to custom; proceeding with existing sort",
            extra={"collection": collection_name},
        )
        COLLECTION_LOGGER.warning(
            "Unable to set Plex collection sort order to custom; proceeding with existing sort",
            extra={"collection": collection_name},
        )
        return False

    @staticmethod
    def _filter_callable_kwargs(func, kwargs: dict) -> dict:
        """Return only keyword arguments accepted by the callable.

        Plex library versions vary in supported parameters. Filtering the kwargs
        prevents passing unsupported keys (e.g., ``collectionSort``) that would
        otherwise raise ``TypeError`` and interrupt collection ordering.
        """

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return kwargs

        filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return filtered if filtered else kwargs

    def _ensure_collection_custom_capabilities(self, collection, collection_name: str, section=None):
        """Ensure a collection supports custom sorting and exposes reordering.

        Some Plex servers do not expose ``reorderItems`` or allow custom sorting
        until after a collection has been edited or reloaded. Because we create
        collections ourselves, proactively configure custom sort flags and try to
        refresh the object so ordering can be applied reliably.
        """

        self._set_collection_custom_sort(collection, collection_name)

        if not callable(getattr(collection, "reorderItems", None)):
            try:
                collection.reload()
            except Exception:
                LOGGER.exception(
                    "Failed to reload Plex collection when enabling custom ordering",
                    extra={"collection": collection_name},
                )
                COLLECTION_LOGGER.exception(
                    "Failed to reload Plex collection when enabling custom ordering",
                    extra={"collection": collection_name},
                )

        if not callable(getattr(collection, "reorderItems", None)) and section is not None:
            try:
                refreshed = section.collection(collection_name)
                if refreshed:
                    self._set_collection_custom_sort(refreshed, collection_name)
                    collection = refreshed
            except Exception:
                LOGGER.exception(
                    "Failed to refresh Plex collection capabilities via section lookup",
                    extra={"collection": collection_name},
                )
                COLLECTION_LOGGER.exception(
                    "Failed to refresh Plex collection capabilities via section lookup",
                    extra={"collection": collection_name},
                )

        if not callable(getattr(collection, "reorderItems", None)):
            LOGGER.debug(
                "Plex collection still missing reorderItems after refresh; ordering may require fallback",
                extra={"collection": collection_name},
            )
            COLLECTION_LOGGER.info(
                "Plex collection still missing reorderItems after refresh; ordering may require fallback",
                extra={"collection": collection_name},
            )

        return collection

    def reorder_collection_items(self, collection, items: Iterable, collection_name: str) -> bool:
        rating_keys = [
            getattr(item, "ratingKey", None)
            for item in items
            if hasattr(item, "ratingKey")
        ]
        rating_keys = [key for key in rating_keys if key is not None]

        if not rating_keys:
            LOGGER.warning(
                "Unable to reorder Plex collection items without rating keys",
                extra={"collection": collection_name},
            )
            COLLECTION_LOGGER.warning(
                "Unable to reorder Plex collection items without rating keys",
                extra={"collection": collection_name},
            )
            return False

        self._set_collection_custom_sort(collection, collection_name)

        reorder_fn = getattr(collection, "reorderItems", None)
        if callable(reorder_fn):
            try:
                reorder_fn(rating_keys)
                LOGGER.debug(
                    "Applied custom Plex collection ordering",
                    extra={
                        "collection": collection_name,
                        "item_count": len(rating_keys),
                    },
                )
                COLLECTION_LOGGER.info(
                    "Applied custom Plex collection ordering",
                    extra={
                        "collection": collection_name,
                        "item_count": len(rating_keys),
                        "rating_keys": rating_keys,
                    },
                )
                return True
            except Exception:
                LOGGER.exception(
                    "Failed to reorder Plex collection items",
                    extra={
                        "collection": collection_name,
                        "item_count": len(rating_keys),
                    },
                )
                COLLECTION_LOGGER.exception(
                    "Failed to reorder Plex collection items",
                    extra={"collection": collection_name, "desired_order": rating_keys},
                )
                return False

        LOGGER.warning(
            "Plex collection object does not support reordering items",
            extra={"collection": collection_name},
        )
        COLLECTION_LOGGER.info(
            "Plex collection object does not support reordering items; relying on insertion order",
            extra={"collection": collection_name, "item_count": len(rating_keys)},
        )
        return False

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
