from __future__ import annotations

import json
import math
import pickle
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from app.config import settings
from app.services.generate_logging import get_generate_logger
from app.services.letterboxd_client import LetterboxdClient, MediaProfile, _tmdb_score
from app.services.plex_service import PlexService, get_plex_service


LOGGER = get_generate_logger()
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = Path("/app/index")
INDEX_PATH = INDEX_DIR / "plex_index.pkl"
CHECKPOINT_PATH = INDEX_DIR / "plex_index.checkpoint.pkl"
PROGRESS_PATH = INDEX_DIR / "plex_index.progress.json"
DEFAULT_STANDUP_CUES = {
    "standup",
    "stand up",
    "stand-up",
    "standup comedy",
    "stand up comedy",
    "stand-up comedy",
    "standup special",
    "stand up special",
    "stand-up special",
    "standup comic",
    "stand-up comic",
}


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


def _format_date(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            return None
    if isinstance(value, str):
        return value
    return None


def _parse_tmdb_id(identifier: str) -> Optional[int]:
    match = re.search(r"themoviedb://(\d+)", identifier) or re.search(r"tmdb(?:\.tv)?://(\d+)", identifier)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    digits = re.findall(r"\d+", identifier)
    if digits:
        try:
            return int(digits[0])
        except ValueError:
            return None
    return None


def _normalize_label(label: str) -> str:
    return re.sub(r"[\s\-]+", " ", label).strip().lower()


def _is_standup_title(genres: Iterable[str], keywords: Iterable[str], letterboxd_keywords: Iterable[str], collections: Iterable[str]) -> bool:
    cues = {_normalize_label(keyword) for keyword in settings.standup_keywords}
    cues.update(DEFAULT_STANDUP_CUES)

    def _matches(values: Iterable[str]) -> bool:
        for value in values or []:
            normalized = _normalize_label(str(value))
            if not normalized:
                continue
            for cue in cues:
                if cue and (normalized == cue or cue in normalized):
                    return True
        return False

    return _matches(genres) or _matches(keywords) or _matches(letterboxd_keywords) or _matches(collections)


@dataclass
class PlexProfile:
    rating_key: int
    title: str
    media_type: str
    cast: Set[str] = field(default_factory=set)
    directors: Set[str] = field(default_factory=set)
    writers: Set[str] = field(default_factory=set)
    genres: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    letterboxd_keywords: Set[str] = field(default_factory=set)
    studios: Set[str] = field(default_factory=set)
    networks: Set[str] = field(default_factory=set)
    number_of_seasons: Optional[int] = None
    number_of_episodes: Optional[int] = None
    episode_run_time: Optional[int] = None
    first_air_date: Optional[str] = None
    collections: Set[str] = field(default_factory=set)
    countries: Set[str] = field(default_factory=set)
    summary: str = ""
    year: Optional[int] = None
    first_air_year: Optional[int] = None
    season_count: Optional[int] = None
    episode_count: Optional[int] = None
    runtime_minutes: Optional[float] = None
    status: Optional[str] = None
    added_at: Optional[datetime] = None
    last_viewed_at: Optional[datetime] = None
    library: Optional[str] = None
    tmdb_id: Optional[int] = None
    tmdb_rating: Optional[float] = None
    letterboxd_rating: Optional[float] = None
    is_standup: bool = False


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
        self._build_started_at: float | None = None
        self._total_items: int | None = None
        self._processed_items: int = 0
        self._tmdb_client: LetterboxdClient | None = None

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._load_from_cache()

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

    def _get_tmdb_client(self) -> LetterboxdClient:
        with self._lock:
            if self._tmdb_client is None:
                self._tmdb_client = LetterboxdClient()
            return self._tmdb_client

    def _tmdb_id_for_item(self, item) -> Optional[int]:
        potential_ids = []
        for guid in getattr(item, "guids", []) or []:
            identifier = getattr(guid, "id", None) or getattr(guid, "guid", None)
            if identifier:
                potential_ids.append(identifier)

        for identifier in potential_ids:
            parsed = _parse_tmdb_id(str(identifier))
            if parsed is not None:
                return parsed

        fallback_guid = getattr(item, "guid", None)
        if fallback_guid:
            parsed = _parse_tmdb_id(str(fallback_guid))
            if parsed is not None:
                return parsed

        for attr in ["tmdb_id", "tmdbId", "tmdbid"]:
            value = getattr(item, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue

        return None

    def _should_fetch_media_profile(self) -> bool:
        return bool(
            settings.tmdb_api_key
            and (
                settings.quality_weight > 0
                or settings.keyword_weight > 0
                or settings.letterboxd_keyword_weight > 0
            )
        )

    def _fetch_media_profile(self, item, media_type: str) -> Optional[MediaProfile]:
        if not self._should_fetch_media_profile():
            return None

        tmdb_id = self._tmdb_id_for_item(item)
        try:
            client = self._get_tmdb_client()
        except Exception:
            LOGGER.exception("Failed to initialize TMDB client for quality scoring")
            return None

        if tmdb_id is None:
            tmdb_id = client.search_tmdb_id(getattr(item, "title", ""), media_type, getattr(item, "year", None))

        if tmdb_id is None:
            return None

        try:
            return client.fetch_profile(tmdb_id, media_type)
        except Exception:
            LOGGER.exception("Failed to fetch TMDB profile for quality scoring", extra={"tmdb_id": tmdb_id})
            return None

    def _build_profile(self, item) -> PlexProfile:
        directors = _extract_names(getattr(item, "directors", []) or [])
        writers = _extract_names(getattr(item, "writers", []) or [])
        cast_sources = getattr(item, "actors", None) or getattr(item, "roles", None) or []
        cast = _extract_names(cast_sources)
        genres = _extract_names(getattr(item, "genres", []) or [])
        studios = _extract_names(getattr(item, "studios", []) or [])
        networks = _extract_names(getattr(item, "networks", []) or [])
        studio_name = getattr(item, "studio", None) or getattr(item, "network", None)
        if studio_name:
            studios.add(str(studio_name))
            networks.add(str(studio_name))
        collections = _extract_names(getattr(item, "collections", []) or [])
        countries = _extract_names(getattr(item, "countries", []) or [])

        summary = _normalize_summary(getattr(item, "summary", ""))
        year = getattr(item, "year", None)
        first_air_date = getattr(item, "originallyAvailableAt", None)
        if hasattr(first_air_date, "year"):
            first_air_year = getattr(first_air_date, "year", None)
        else:
            first_air_year = None
        if first_air_year is None:
            first_air_year = year
        season_count = getattr(item, "childCount", None)
        if season_count is not None:
            try:
                season_count = int(season_count)
            except (TypeError, ValueError):
                season_count = None
        episode_count = getattr(item, "leafCount", None)
        if episode_count is not None:
            try:
                episode_count = int(episode_count)
            except (TypeError, ValueError):
                episode_count = None
        duration_ms = getattr(item, "duration", None)
        runtime_minutes = None
        if duration_ms:
            try:
                runtime_minutes = float(duration_ms) / 60000.0
            except (TypeError, ValueError):
                runtime_minutes = None
        status = getattr(item, "status", None)
        added_at = getattr(item, "addedAt", None)
        last_viewed_at = getattr(item, "lastViewedAt", None)
        media_type = getattr(item, "type", "movie") or "movie"

        metadata = self._fetch_media_profile(item, media_type)
        tmdb_rating = metadata.tmdb_rating if metadata else None
        letterboxd_rating = metadata.letterboxd_rating if metadata else None
        keywords = set(metadata.keywords) if metadata else set()
        letterboxd_keywords = set(metadata.letterboxd_keywords) if metadata else set()
        tmdb_id = metadata.tmdb_id if metadata else self._tmdb_id_for_item(item)
        if metadata and metadata.networks:
            networks.update(metadata.networks)
        number_of_seasons = metadata.number_of_seasons if metadata else None
        if number_of_seasons is None:
            number_of_seasons = season_count
        number_of_episodes = metadata.number_of_episodes if metadata else None
        if number_of_episodes is None:
            number_of_episodes = episode_count
        episode_run_time = metadata.episode_run_time if metadata else None
        if episode_run_time is None and runtime_minutes is not None:
            try:
                episode_run_time = int(round(runtime_minutes))
            except (TypeError, ValueError):
                episode_run_time = None
        first_air_date_value = metadata.first_air_date if metadata else None
        if first_air_date_value is None:
            first_air_date_value = _format_date(first_air_date)
        is_standup = _is_standup_title(genres, keywords, letterboxd_keywords, collections)

        return PlexProfile(
            rating_key=int(getattr(item, "ratingKey", 0)),
            title=getattr(item, "title", "Unknown"),
            media_type=media_type,
            cast=cast,
            directors=directors,
            writers=writers,
            genres=genres,
            keywords=keywords,
            letterboxd_keywords=letterboxd_keywords,
            studios=studios,
            networks=networks,
            number_of_seasons=number_of_seasons,
            number_of_episodes=number_of_episodes,
            episode_run_time=episode_run_time,
            first_air_date=first_air_date_value,
            collections=collections,
            countries=countries,
            summary=summary,
            year=year,
            first_air_year=first_air_year,
            season_count=season_count,
            episode_count=episode_count,
            runtime_minutes=runtime_minutes,
            status=status,
            added_at=added_at,
            last_viewed_at=last_viewed_at,
            library=getattr(item, "librarySectionTitle", None),
            tmdb_id=tmdb_id,
            tmdb_rating=tmdb_rating,
            letterboxd_rating=letterboxd_rating,
            is_standup=is_standup,
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

    @staticmethod
    def _ensure_profile_defaults(profile: PlexProfile) -> PlexProfile:
        if not hasattr(profile, "cast") or profile.cast is None:
            profile.cast = set()
        if not hasattr(profile, "directors") or profile.directors is None:
            profile.directors = set()
        if not hasattr(profile, "writers") or profile.writers is None:
            profile.writers = set()
        if not hasattr(profile, "genres") or profile.genres is None:
            profile.genres = set()
        if not hasattr(profile, "studios") or profile.studios is None:
            profile.studios = set()
        if not hasattr(profile, "networks") or profile.networks is None:
            profile.networks = set()
        if not hasattr(profile, "number_of_seasons"):
            profile.number_of_seasons = None
        if not hasattr(profile, "number_of_episodes"):
            profile.number_of_episodes = None
        if not hasattr(profile, "episode_run_time"):
            profile.episode_run_time = None
        if not hasattr(profile, "first_air_date"):
            profile.first_air_date = None
        if not hasattr(profile, "collections") or profile.collections is None:
            profile.collections = set()
        if not hasattr(profile, "countries") or profile.countries is None:
            profile.countries = set()
        if not hasattr(profile, "keywords") or profile.keywords is None:
            profile.keywords = set()
        if not hasattr(profile, "letterboxd_keywords") or profile.letterboxd_keywords is None:
            profile.letterboxd_keywords = set()
        if not hasattr(profile, "tmdb_id"):
            profile.tmdb_id = None
        if not hasattr(profile, "tmdb_rating"):
            profile.tmdb_rating = None
        if not hasattr(profile, "letterboxd_rating"):
            profile.letterboxd_rating = None
        if not hasattr(profile, "summary") or profile.summary is None:
            profile.summary = ""
        if not hasattr(profile, "year"):
            profile.year = None
        if not hasattr(profile, "first_air_year"):
            profile.first_air_year = None
        if not hasattr(profile, "season_count"):
            profile.season_count = None
        if not hasattr(profile, "episode_count"):
            profile.episode_count = None
        if not hasattr(profile, "runtime_minutes"):
            profile.runtime_minutes = None
        if not hasattr(profile, "status"):
            profile.status = None
        if not hasattr(profile, "added_at"):
            profile.added_at = None
        if not hasattr(profile, "last_viewed_at"):
            profile.last_viewed_at = None
        if not hasattr(profile, "library"):
            profile.library = None
        existing_flag = getattr(profile, "is_standup", False)
        try:
            profile.is_standup = bool(
                existing_flag
                or _is_standup_title(
                    getattr(profile, "genres", set()),
                    getattr(profile, "keywords", set()),
                    getattr(profile, "letterboxd_keywords", set()),
                    getattr(profile, "collections", set()),
                )
            )
        except Exception:
            LOGGER.exception("Failed to evaluate stand-up label for profile", extra={"rating_key": getattr(profile, "rating_key", None)})
            profile.is_standup = bool(existing_flag)
        return profile

    def _update_latest_added(self, profile: PlexProfile) -> None:
        if not profile.added_at:
            return
        current = self._latest_added.get(profile.media_type)
        if current is None or profile.added_at > current:
            self._latest_added[profile.media_type] = profile.added_at

    def _serialize_checkpoint(self, profiles: Dict[int, PlexProfile], latest: dict[str, datetime]):
        data = {
            "profiles": profiles,
            "latest_added": latest,
            "processed_items": len(profiles),
            "created_at": time.time(),
        }
        try:
            with CHECKPOINT_PATH.open("wb") as fp:
                pickle.dump(data, fp)
            PROGRESS_PATH.write_text(
                json.dumps(
                    {
                        "state": "building",
                        "processed_items": len(profiles),
                        "total_items": self._total_items,
                        "started_at": self._build_started_at,
                        "last_updated_at": time.time(),
                    }
                )
            )
        except Exception:
            LOGGER.exception("Failed to persist Plex index checkpoint")

    def _load_checkpoint(self) -> tuple[Dict[int, PlexProfile], dict[str, datetime]]:
        if not CHECKPOINT_PATH.exists():
            return {}, {}
        try:
            with CHECKPOINT_PATH.open("rb") as fp:
                data = pickle.load(fp)
            profiles = data.get("profiles", {})
            profiles = {key: self._ensure_profile_defaults(profile) for key, profile in profiles.items()}
            latest = data.get("latest_added", {})
            return profiles, latest
        except Exception:
            LOGGER.exception("Failed to load Plex index checkpoint")
            return {}, {}

    def _save_cache(self) -> None:
        data = {
            "profiles": self._profiles,
            "latest_added": self._latest_added,
            "summary_matrix": self._summary_matrix,
            "matrix_keys": self._matrix_keys,
            "last_built_at": self._last_built_at,
        }
        try:
            with INDEX_PATH.open("wb") as fp:
                pickle.dump(data, fp)
        except Exception:
            LOGGER.exception("Failed to persist Plex index cache")

    def _load_from_cache(self) -> None:
        if not INDEX_PATH.exists():
            return
        try:
            with INDEX_PATH.open("rb") as fp:
                data = pickle.load(fp)
            self._profiles = {
                key: self._ensure_profile_defaults(profile) for key, profile in data.get("profiles", {}).items()
            }
            self._latest_added = data.get("latest_added", {})
            self._summary_matrix = data.get("summary_matrix")
            self._matrix_keys = data.get("matrix_keys", [])
            self._has_built = bool(self._profiles)
            self._last_built_at = data.get("last_built_at")
            self._processed_items = len(self._profiles)
            self._total_items = len(self._profiles) or None
        except Exception:
            LOGGER.exception("Failed to load Plex index cache")

    def rebuild(self) -> None:
        with self._lock:
            if self._build_in_progress:
                LOGGER.info("Plex index rebuild already in progress")
                return
            self._build_in_progress = True
            self._last_error = None
            self._last_started_at = datetime.utcnow()
            self._build_started_at = time.time()
            self._processed_items = 0
            self._total_items = None

        try:
            checkpoint_profiles, checkpoint_latest = self._load_checkpoint()
            profiles: Dict[int, PlexProfile] = dict(checkpoint_profiles)
            latest: dict[str, datetime] = dict(checkpoint_latest)
            processed_keys = set(profiles.keys())
            self._processed_items = len(profiles)

            def _process_item(item):
                profile = self._build_profile(item)
                profiles[profile.rating_key] = profile
                if profile.added_at:
                    latest[profile.media_type] = max(
                        latest.get(profile.media_type, profile.added_at), profile.added_at
                    )

            total_items = self._processed_items

            for item in self.plex.iter_library_items("movie"):
                total_items += 1
                self._total_items = total_items
                rating_key = getattr(item, "ratingKey", None)
                if rating_key is not None and rating_key in processed_keys:
                    continue
                _process_item(item)
                self._processed_items += 1
                if self._processed_items % 50 == 0:
                    self._serialize_checkpoint(profiles, latest)

            for item in self.plex.iter_library_items("show"):
                total_items += 1
                self._total_items = total_items
                rating_key = getattr(item, "ratingKey", None)
                if rating_key is not None and rating_key in processed_keys:
                    continue
                _process_item(item)
                self._processed_items += 1
                if self._processed_items % 50 == 0:
                    self._serialize_checkpoint(profiles, latest)

            self._total_items = total_items or self._total_items
            self._serialize_checkpoint(profiles, latest)

            with self._lock:
                self._profiles = profiles
                self._latest_added = latest
                self._rebuild_text_matrix()
                self._build_in_progress = False
                self._has_built = True
                self._last_built_at = datetime.utcnow()
                self._processed_items = len(profiles)
            self._save_cache()
            if CHECKPOINT_PATH.exists():
                CHECKPOINT_PATH.unlink(missing_ok=True)
            if PROGRESS_PATH.exists():
                PROGRESS_PATH.unlink(missing_ok=True)
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

        raw_scores = np.array(scores, dtype=float)
        if raw_scores.size == 0:
            return {}

        clipped_scores = np.clip(raw_scores, -1.0, 1.0)

        return {key: float(score) for key, score in zip(keys, clipped_scores)}

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

            progress = None
            eta_seconds = None
            if self._total_items and self._total_items > 0:
                progress = min(self._processed_items / self._total_items, 1.0)
                if self._build_started_at and self._processed_items:
                    elapsed = time.time() - self._build_started_at
                    rate = elapsed / max(self._processed_items, 1)
                    eta_seconds = max(int((self._total_items - self._processed_items) * rate), 0)

            return {
                "state": state,
                "items_indexed": len(self._profiles),
                "last_started_at": self._last_started_at.isoformat() if self._last_started_at else None,
                "last_built_at": self._last_built_at.isoformat() if self._last_built_at else None,
                "last_error": self._last_error,
                "total_items": self._total_items,
                "processed_items": self._processed_items,
                "progress": progress,
                "eta_seconds": eta_seconds,
            }


def _overlap_ratio(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)

def _numeric_similarity(source_value: Optional[float], target_value: Optional[float]) -> float:
    if source_value is None or target_value is None:
        return 0.0
    try:
        source = float(source_value)
        target = float(target_value)
    except (TypeError, ValueError):
        return 0.0
    if source <= 0 or target <= 0:
        return 0.0
    max_value = max(source, target)
    if max_value <= 0:
        return 0.0
    similarity = 1.0 - abs(source - target) / max_value
    return max(0.0, min(1.0, similarity))


def _status_similarity(source_status: Optional[str], target_status: Optional[str]) -> float:
    if not source_status or not target_status:
        return 0.0
    return 1.0 if _normalize_label(str(source_status)) == _normalize_label(str(target_status)) else 0.0


def _year_similarity(source_year: Optional[int], target_year: Optional[int], half_life: float) -> float:
    if not source_year or not target_year:
        return 0.0
    year_gap = abs(source_year - target_year)
    if half_life <= 0:
        return 0.0
    return math.exp(-year_gap / half_life)

def _recency_bonus(profile: PlexProfile, max_bonus: float, half_life_days: float) -> float:
    recent_date = max(
        [dt for dt in [profile.added_at, profile.last_viewed_at] if dt is not None],
        default=None,
    )
    if recent_date is None:
        return 0.0
    age_days = (datetime.utcnow() - recent_date).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0
    if half_life_days <= 0:
        return 0.0
    decay = math.exp(-age_days / half_life_days)
    return max_bonus * decay


def _quality_score(profile: PlexProfile) -> float:
    rating_value: Optional[float] = profile.tmdb_rating
    if rating_value is None and profile.letterboxd_rating is not None:
        rating_value = profile.letterboxd_rating * 20.0

    base_score = _tmdb_score(rating_value) if rating_value is not None else 0.0
    normalized = base_score / 50.0
    return settings.quality_weight * normalized


def profile_similarity(source: PlexProfile, target: PlexProfile, plot_score: float = 0.0) -> Tuple[float, dict[str, float]]:
    if source.media_type == "show" and target.media_type == "show":
        genre_similarity = _overlap_ratio(source.genres, target.genres)
        network_similarity = _overlap_ratio(source.networks, target.networks)
        season_similarity = _numeric_similarity(source.season_count, target.season_count)
        episode_similarity = _numeric_similarity(source.episode_count, target.episode_count)
        runtime_similarity = _numeric_similarity(source.runtime_minutes, target.runtime_minutes)
        status_similarity = _status_similarity(source.status, target.status)
        year_similarity = _year_similarity(source.first_air_year, target.first_air_year, settings.show_year_half_life)
        recency = _recency_bonus(target, settings.show_recency_max_bonus, settings.show_recency_half_life_days)
        quality = _quality_score(target)

        breakdown = {
            "show_genres": genre_similarity * settings.show_genre_weight,
            "show_networks": network_similarity * settings.show_network_weight,
            "show_season_count": season_similarity * settings.show_season_count_weight,
            "show_episode_count": episode_similarity * settings.show_episode_count_weight,
            "show_runtime": runtime_similarity * settings.show_runtime_weight,
            "show_status": status_similarity * settings.show_status_weight,
            "show_plot": plot_score * settings.plot_weight,
            "show_year": year_similarity * settings.show_year_weight,
            "show_recency": recency,
            "show_quality": quality,
        }
    else:
        cast_similarity = _overlap_ratio(source.cast, target.cast)
        director_similarity = _overlap_ratio(source.directors, target.directors)
        writer_similarity = _overlap_ratio(source.writers, target.writers)
        genre_similarity = _overlap_ratio(source.genres, target.genres)
        keyword_similarity = _overlap_ratio(source.keywords, target.keywords)
        letterboxd_keyword_similarity = _overlap_ratio(source.letterboxd_keywords, target.letterboxd_keywords)
        studio_similarity = _overlap_ratio(source.studios, target.studios)
        collection_similarity = _overlap_ratio(source.collections, target.collections)
        country_similarity = _overlap_ratio(source.countries, target.countries)
        year_similarity = _year_similarity(source.year, target.year, settings.year_half_life)
        recency = _recency_bonus(target, settings.recency_max_bonus, settings.recency_half_life_days)
        quality = _quality_score(target)

        breakdown = {
            "cast": cast_similarity * settings.cast_weight,
            "directors": director_similarity * settings.director_weight,
            "writers": writer_similarity * settings.writer_weight,
            "genres": genre_similarity * settings.genre_weight,
            "keywords": keyword_similarity * settings.keyword_weight,
            "letterboxd_keywords": letterboxd_keyword_similarity * settings.letterboxd_keyword_weight,
            "studios": studio_similarity * settings.studio_weight,
            "collections": collection_similarity * settings.collection_weight,
            "countries": country_similarity * settings.country_weight,
            "plot": plot_score * settings.plot_weight,
            "year": year_similarity * settings.year_weight,
            "recency": recency,
            "quality": quality,
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
