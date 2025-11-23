from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from app.services.generate_logging import get_generate_logger
from app.services.plex_service import PlexService, get_plex_service


LOGGER = get_generate_logger()
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _extract_names(entries: Iterable) -> Set[str]:
    names: Set[str] = set()
    for entry in entries or []:
        name = getattr(entry, "tag", None) or getattr(entry, "name", None)
        if not name:
            continue
        names.add(str(name))
    return names


def _normalize_summary(summary: Optional[str]) -> str:
    return (summary or "").strip()


@dataclass
class PlexProfile:
    rating_key: int
    title: str
    media_type: str
    cast: Set[str] = field(default_factory=set)
    directors: Set[str] = field(default_factory=set)
    writers: Set[str] = field(default_factory=set)
    genres: Set[str] = field(default_factory=set)
    summary: str = ""
    year: Optional[int] = None
    added_at: Optional[datetime] = None
    library: Optional[str] = None


class PlexIndex:
    def __init__(self, plex: PlexService) -> None:
        self.plex = plex
        self._profiles: Dict[int, PlexProfile] = {}
        self._embedding_model = None
        self._summary_matrix: np.ndarray | None = None
        self._lock = threading.RLock()
        self._latest_added: dict[str, datetime] = {}
        self._matrix_keys: list[int] = []
        self._build_in_progress = False
        self._has_built = False
        self._last_built_at: datetime | None = None
        self._last_started_at: datetime | None = None
        self._last_error: str | None = None

    def _get_embedding_model(self):
        with self._lock:
            if self._embedding_model is not None:
                return self._embedding_model

            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to load embedding model")
                raise

            return self._embedding_model

    def _encode_summaries(self, summaries: list[str]) -> np.ndarray:
        model = self._get_embedding_model()
        return model.encode(
            summaries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _build_profile(self, item) -> PlexProfile:
        directors = _extract_names(getattr(item, "directors", []) or [])
        writers = _extract_names(getattr(item, "writers", []) or [])
        cast_sources = getattr(item, "actors", None) or getattr(item, "roles", None) or []
        cast = _extract_names(cast_sources)
        genres = _extract_names(getattr(item, "genres", []) or [])

        summary = _normalize_summary(getattr(item, "summary", ""))
        year = getattr(item, "year", None)
        added_at = getattr(item, "addedAt", None)
        media_type = getattr(item, "type", "movie") or "movie"

        return PlexProfile(
            rating_key=int(getattr(item, "ratingKey", 0)),
            title=getattr(item, "title", "Unknown"),
            media_type=media_type,
            cast=cast,
            directors=directors,
            writers=writers,
            genres=genres,
            summary=summary,
            year=year,
            added_at=added_at,
            library=getattr(item, "librarySectionTitle", None),
        )

    def _rebuild_text_matrix(self) -> None:
        summaries: List[str] = []
        keys: List[int] = []
        for key, profile in self._profiles.items():
            if profile.summary:
                summaries.append(profile.summary)
                keys.append(key)

        if not summaries:
            self._summary_matrix = None
            self._matrix_keys = []
            return

        try:
            self._summary_matrix = self._encode_summaries(summaries)
            self._matrix_keys = keys
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to build summary embeddings")
            self._summary_matrix = None
            self._matrix_keys = []

    def _update_latest_added(self, profile: PlexProfile) -> None:
        if not profile.added_at:
            return
        current = self._latest_added.get(profile.media_type)
        if current is None or profile.added_at > current:
            self._latest_added[profile.media_type] = profile.added_at

    def rebuild(self) -> None:
        with self._lock:
            if self._build_in_progress:
                LOGGER.info("Plex index rebuild already in progress")
                return
            self._build_in_progress = True
            self._last_error = None
            self._last_started_at = datetime.utcnow()

        try:
            profiles: Dict[int, PlexProfile] = {}
            latest: dict[str, datetime] = {}
            for item in self.plex.iter_library_items("movie"):
                profile = self._build_profile(item)
                profiles[profile.rating_key] = profile
                if profile.added_at:
                    latest["movie"] = max(latest.get("movie", profile.added_at), profile.added_at)

            for item in self.plex.iter_library_items("show"):
                profile = self._build_profile(item)
                profiles[profile.rating_key] = profile
                if profile.added_at:
                    latest["show"] = max(latest.get("show", profile.added_at), profile.added_at)

            with self._lock:
                self._profiles = profiles
                self._latest_added = latest
                self._rebuild_text_matrix()
                self._build_in_progress = False
                self._has_built = True
                self._last_built_at = datetime.utcnow()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._build_in_progress = False
                self._last_error = str(exc)
            raise

        LOGGER.info(
            "Rebuilt Plex metadata index",
            extra={"count": len(self._profiles)},
        )

    def refresh_recent_additions(self, limit: int = 50) -> int:
        """Pull recently added items and update the index incrementally."""

        additions: List[PlexProfile] = []
        for media_type in ("movie", "show"):
            newest_seen = self._latest_added.get(media_type, datetime.min)
            for item in self.plex.recently_added(media_type=media_type, max_results=limit):
                added_at = getattr(item, "addedAt", None) or datetime.min
                rating_key = getattr(item, "ratingKey", None)
                if rating_key is None:
                    continue
                if rating_key in self._profiles and added_at <= newest_seen:
                    continue
                profile = self._build_profile(item)
                additions.append(profile)

        if not additions:
            return 0

        with self._lock:
            for profile in additions:
                self._profiles[profile.rating_key] = profile
                self._update_latest_added(profile)
            self._rebuild_text_matrix()
            self._has_built = True
            self._last_built_at = datetime.utcnow()

        LOGGER.info(
            "Indexed %s new Plex items", len(additions)
        )
        return len(additions)

    def profile_for_item(self, item) -> Optional[PlexProfile]:
        if item is None:
            return None
        rating_key = getattr(item, "ratingKey", None)
        if rating_key is None:
            return None
        with self._lock:
            profile = self._profiles.get(int(rating_key))
            if profile:
                return profile
        # If not already indexed, build a profile on the fly without mutating the index
        try:
            return self._build_profile(item)
        except Exception:
            LOGGER.exception("Failed to build profile for Plex item", extra={"rating_key": rating_key})
            return None

    def _plot_similarity_scores(self, source: PlexProfile) -> dict[int, float]:
        with self._lock:
            matrix = self._summary_matrix
            keys = list(self._matrix_keys)

        if matrix is None or not keys:
            return {}
        if not source.summary:
            return {}

        try:
            source_embedding = self._encode_summaries([source.summary])[0]
            scores = np.matmul(matrix, source_embedding)
        except Exception:
            LOGGER.exception("Failed to compute plot similarity")
            return {}

        return {key: float(score) for key, score in zip(keys, scores)}

    def related_profiles(
        self, source: PlexProfile, limit: Optional[int] = None
    ) -> List[PlexProfile]:
        plot_scores = self._plot_similarity_scores(source)
        candidates: List[PlexProfile] = []
        with self._lock:
            for profile in self._profiles.values():
                if profile.rating_key == source.rating_key:
                    continue
                if profile.media_type != source.media_type:
                    continue
                if limit is not None and len(candidates) >= limit:
                    break
                candidates.append(profile)
        if plot_scores:
            candidates.sort(key=lambda p: plot_scores.get(p.rating_key, 0.0), reverse=True)
        return candidates

    def status(self) -> dict[str, object]:
        with self._lock:
            if self._build_in_progress:
                state = "building"
            elif self._has_built:
                state = "ready"
            else:
                state = "idle"

            return {
                "state": state,
                "items_indexed": len(self._profiles),
                "last_started_at": self._last_started_at.isoformat() if self._last_started_at else None,
                "last_built_at": self._last_built_at.isoformat() if self._last_built_at else None,
                "last_error": self._last_error,
            }


def profile_similarity(source: PlexProfile, target: PlexProfile, plot_score: float = 0.0) -> Tuple[float, dict[str, float]]:
    breakdown = {
        "cast": 10.0 * len(source.cast & target.cast),
        "directors": 30.0 * len(source.directors & target.directors),
        "writers": 20.0 * len(source.writers & target.writers),
        "genres": 10.0 * len(source.genres & target.genres),
        "plot": round(plot_score * 50.0, 2),
    }
    total = round(sum(breakdown.values()), 2)
    normalized = {key: round(value, 2) for key, value in breakdown.items()}
    return total, normalized


_INDEX: PlexIndex | None = None
_INDEX_LOCK = threading.Lock()


def get_plex_index() -> PlexIndex:
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            plex = get_plex_service()
            _INDEX = PlexIndex(plex)
        return _INDEX
