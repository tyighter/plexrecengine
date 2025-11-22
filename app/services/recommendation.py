from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from app.services.generate_logging import get_generate_logger
from app.services.letterboxd_client import LetterboxdClient, MediaProfile, profile_similarity
from app.services.plex_service import PlexService


@dataclass
class Recommendation:
    title: str
    score: float
    poster: str | None
    rating_key: int
    letterboxd_rating: float | None = None
    source_title: str | None = None
    reason: str | None = None
    score_breakdown: dict[str, float] | None = None


LOGGER = get_generate_logger()


class RecommendationEngine:
    def __init__(self, plex: PlexService, letterboxd: LetterboxdClient) -> None:
        self.plex = plex
        self.letterboxd = letterboxd

    def _profile_for_item(self, item, media_type: str) -> Optional[MediaProfile]:
        guid = item.guid or ""
        tmdb_prefix = "tmdb://"
        tmdb_id = None
        if guid.startswith(tmdb_prefix):
            try:
                tmdb_id = int(guid[len(tmdb_prefix) :].split("?")[0])
            except ValueError:
                pass
        if tmdb_id is None:
            tmdb_id = self.letterboxd.search_tmdb_id(item.title, media_type, getattr(item, "year", None))
        if tmdb_id is None:
            return None
        return self.letterboxd.fetch_profile(tmdb_id, media_type)

    def _score_related(
        self, source_profile: MediaProfile, candidates: Iterable[MediaProfile]
    ) -> List[Tuple[MediaProfile, float, dict[str, float]]]:
        scored = []
        for candidate in candidates:
            if candidate.tmdb_id == source_profile.tmdb_id:
                continue
            score, breakdown = profile_similarity(source_profile, candidate)
            if score > 0:
                scored.append((candidate, score, breakdown))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def top_recommendations_for_item(
        self, item, media_type: str, count: int = 3
    ) -> List[Recommendation]:
        source_profile = self._profile_for_item(item, media_type)
        if source_profile is None:
            return []
        related = self.letterboxd.search_related(source_profile, limit=20)
        scored = self._score_related(source_profile, related)
        source_title = getattr(item, "title", None)
        source_year = getattr(item, "year", None)
        source_label = (
            f"{source_title} ({source_year})" if source_title and source_year else source_title
        )
        recommendations: List[Recommendation] = []
        for profile, score, breakdown in scored[:count * 2]:
            # try to find an unwatched matching item in Plex
            for plex_item in self.plex.search_unwatched(section_type=media_type, query=profile.title):
                reason_parts = [
                    "Recommended because you recently watched",
                    source_label or "a similar title",
                ]
                if profile.letterboxd_rating:
                    reason_parts.append(
                        f"and it pairs well with {profile.title} (Letterboxd {profile.letterboxd_rating:.1f})"
                    )
                else:
                    reason_parts.append(f"and it pairs well with {profile.title}")
                reason_parts.append(f"Similarity score: {score:.2f}")
                recommendations.append(
                    Recommendation(
                        title=plex_item.title,
                        score=score,
                        poster=self.plex.poster_url(plex_item),
                        rating_key=plex_item.ratingKey,
                        letterboxd_rating=profile.letterboxd_rating,
                        source_title=source_title,
                        reason=". ".join(reason_parts),
                        score_breakdown=breakdown,
                    )
                )
                break
            if len(recommendations) >= count:
                break
        return recommendations

    def _refresh_collection(self, collection_name: str, recs: List[Recommendation]):
        items = []
        for rec in recs:
            item = self.plex.fetch_item(
                rec.rating_key,
                extra={
                    "source": "recommendation_collection",
                    "collection": collection_name,
                    "title": rec.title,
                    "score": rec.score,
                },
            )
            if item:
                items.append(item)
        try:
            self.plex.set_collection_members(collection_name, items)
        except Exception:
            LOGGER.exception("Failed to refresh Plex collection", extra={"collection": collection_name})

    def build_movie_collection(self, limit: int = 10, days: int = 30) -> List[Recommendation]:
        recommendations: List[Recommendation] = []
        recent_movies = list(self.plex.recently_watched_movies(days=days, max_results=200))
        LOGGER.debug("Found recently watched movies", extra={"count": len(recent_movies)})
        for movie in recent_movies:
            recommendations.extend(self.top_recommendations_for_item(movie, media_type="movie"))
        unique: List[Recommendation] = []
        seen = set()
        for rec in sorted(recommendations, key=lambda r: r.score, reverse=True):
            if rec.rating_key in seen:
                continue
            seen.add(rec.rating_key)
            unique.append(rec)
        final = unique[: limit * 3]
        LOGGER.debug(
            "Compiled movie recommendations",
            extra={"candidates": len(recommendations), "unique": len(final)},
        )
        self._refresh_collection("Recommended Movies", final)
        return final

    def build_show_collection(self, limit: int = 10, days: int = 30) -> List[Recommendation]:
        recommendations: List[Recommendation] = []
        recent_shows = list(self.plex.recently_watched_shows(days=days, max_results=200))
        LOGGER.debug("Found recently watched shows", extra={"count": len(recent_shows)})
        for show in recent_shows:
            recommendations.extend(self.top_recommendations_for_item(show, media_type="tv"))
        unique: List[Recommendation] = []
        seen = set()
        for rec in sorted(recommendations, key=lambda r: r.score, reverse=True):
            if rec.rating_key in seen:
                continue
            seen.add(rec.rating_key)
            unique.append(rec)
        final = unique[: limit * 3]
        LOGGER.debug(
            "Compiled show recommendations",
            extra={"candidates": len(recommendations), "unique": len(final)},
        )
        self._refresh_collection("Recommended Shows", final)
        return final
