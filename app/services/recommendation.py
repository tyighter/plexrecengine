from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from app.services.generate_logging import get_generate_logger, get_scoring_logger
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
    commonalities: dict[str, list[str]] | None = None


LOGGER = get_generate_logger()
SCORING_LOGGER = get_scoring_logger()


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
        self, item, media_type: str, count: int = 6
    ) -> List[Recommendation]:
        source_title = getattr(item, "title", None)
        source_year = getattr(item, "year", None)
        source_label = (
            f"{source_title} ({source_year})" if source_title and source_year else source_title
        )

        SCORING_LOGGER.info("==== Recently watched: %s ====", source_label or "Unknown")
        source_profile = self._profile_for_item(item, media_type)
        if source_profile is None:
            SCORING_LOGGER.info("No profile available; skipping scoring for this item")
            return []
        related = list(self.letterboxd.search_related(source_profile, limit=20))
        SCORING_LOGGER.info(
            "Initial related pool (%s): %s",
            len(related),
            ", ".join(
                f"{candidate.title} (tmdb_id={candidate.tmdb_id})" for candidate in related
            ),
        )
        scored = self._score_related(source_profile, related)
        if not scored:
            SCORING_LOGGER.info("No related titles produced non-zero scores")
        else:
            for rank, (profile, score, breakdown) in enumerate(scored, start=1):
                SCORING_LOGGER.info(
                    "#%s %s (tmdb_id=%s) score=%.2f breakdown=%s",
                    rank,
                    profile.title,
                    profile.tmdb_id,
                    score,
                    breakdown,
                )
        recommendations: List[Recommendation] = []
        skipped_not_in_library: list[str] = []
        for profile, score, breakdown in scored[: count * 2]:
            # try to find an unwatched matching item in Plex
            plex_matches = list(
                self.plex.search_unwatched(section_type=media_type, query=profile.title)
            )
            if not plex_matches:
                skipped_not_in_library.append(profile.title)
                SCORING_LOGGER.info(
                    "Skipping %s (tmdb_id=%s) — not found as unwatched in Plex library",
                    profile.title,
                    profile.tmdb_id,
                )
                continue

            plex_item = plex_matches[0]
            overlap = {
                "directors": sorted(source_profile.directors & profile.directors),
                "writers": sorted(source_profile.writers & profile.writers),
                "cast": sorted(source_profile.cast & profile.cast),
                "genres": sorted(source_profile.genres & profile.genres),
                "keywords": sorted(source_profile.keywords & profile.keywords),
            }

            def describe(values: list[str], label: str) -> str:
                if not values:
                    return ""
                if len(values) == 1:
                    return f"{label} {values[0]}"
                sample = values[:3]
                extra = len(values) - len(sample)
                sample_text = ", ".join(sample)
                if extra > 0:
                    return f"{label} {sample_text} (+{extra} more)"
                return f"{label} {sample_text}"

            reason_parts = [
                f"Recommended because you recently watched {source_label or 'a similar title'}",
            ]
            if profile.letterboxd_rating:
                reason_parts.append(
                    f"It pairs well with {profile.title} (Letterboxd {profile.letterboxd_rating:.1f})"
                )
            else:
                reason_parts.append(f"It pairs well with {profile.title}")

            shared_traits = [
                describe(overlap["directors"], "shares director"),
                describe(overlap["cast"], "features"),
                describe(overlap["writers"], "writer"),
                describe(overlap["genres"], "genre"),
                describe(overlap["keywords"], "keyword"),
            ]
            shared_traits = [part for part in shared_traits if part]
            if shared_traits:
                reason_parts.append("Common threads: " + "; ".join(shared_traits))

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
                    commonalities={
                        key: value for key, value in overlap.items() if value
                    },
                )
            )
            SCORING_LOGGER.info(
                "Selected Plex item '%s' for recommendation (score=%.2f) breakdown=%s",
                plex_item.title,
                score,
                breakdown,
            )
            if len(recommendations) >= count:
                break

        if skipped_not_in_library:
            SCORING_LOGGER.info(
                "Removed %s candidates not present in Plex: %s",
                len(skipped_not_in_library),
                ", ".join(skipped_not_in_library),
            )

        if recommendations:
            top_selected = sorted(recommendations, key=lambda rec: rec.score, reverse=True)[:3]
            SCORING_LOGGER.info(
                "Top %s selected Plex matches: %s",
                len(top_selected),
                ", ".join(
                    f"{rec.title} (score={rec.score:.2f})" for rec in top_selected
                ),
            )
        return recommendations

    def _resolve_conflicts(
        self,
        candidates_by_source: dict[int, List[Recommendation]],
        picks_per_source: int,
    ) -> dict[int, List[Recommendation]]:
        selected: dict[int, List[Recommendation]] = {}
        remaining: dict[int, List[Recommendation]] = {}
        for source, candidates in candidates_by_source.items():
            selected[source] = list(candidates[:picks_per_source])
            remaining[source] = list(candidates[picks_per_source:])

        changed = True
        while changed:
            changed = False
            rating_owner: dict[int, Tuple[int, Recommendation]] = {}

            for source, recs in selected.items():
                for rec in list(recs):
                    owner = rating_owner.get(rec.rating_key)
                    if owner is None or rec.score > owner[1].score:
                        if owner is not None and owner[1] in selected.get(owner[0], []):
                            selected[owner[0]].remove(owner[1])
                            remaining[owner[0]].insert(0, owner[1])
                            changed = True
                        rating_owner[rec.rating_key] = (source, rec)
                    else:
                        selected[source].remove(rec)
                        remaining[source].insert(0, rec)
                        changed = True

            for source in list(selected.keys()):
                while len(selected[source]) < picks_per_source and remaining[source]:
                    candidate = remaining[source].pop(0)
                    owner = rating_owner.get(candidate.rating_key)
                    if owner is None:
                        selected[source].append(candidate)
                        rating_owner[candidate.rating_key] = (source, candidate)
                        changed = True
                    elif candidate.score > owner[1].score:
                        prev_source, prev_rec = owner
                        if prev_rec in selected.get(prev_source, []):
                            selected[prev_source].remove(prev_rec)
                            remaining[prev_source].insert(0, prev_rec)
                        selected[source].append(candidate)
                        rating_owner[candidate.rating_key] = (source, candidate)
                        changed = True

        return selected

    def _build_collection(
        self, recent_items: List, media_type: str, collection_name: str, limit: int
    ) -> List[Recommendation]:
        picks_per_source = 3
        candidates_by_source: dict[int, List[Recommendation]] = {}
        source_order: List[int] = []
        for item in recent_items:
            source_id = getattr(item, "ratingKey", id(item))
            recs = self.top_recommendations_for_item(
                item, media_type=media_type, count=picks_per_source * 3
            )
            if not recs:
                continue
            candidates_by_source[source_id] = recs
            source_order.append(source_id)

        selected_by_source = self._resolve_conflicts(candidates_by_source, picks_per_source)

        final: List[Recommendation] = []
        for source in source_order:
            source_recs = sorted(
                selected_by_source.get(source, []), key=lambda rec: rec.score, reverse=True
            )
            final.extend(source_recs)
            if len(final) >= limit * picks_per_source:
                break

        final = final[: limit * picks_per_source]
        LOGGER.debug(
            "Compiled %s recommendations", collection_name,
            extra={"sources": len(source_order), "final": len(final)},
        )
        self._refresh_collection(collection_name, final)
        return final

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
        recent_movies = list(self.plex.recently_watched_movies(days=days, max_results=200))
        LOGGER.debug("Found recently watched movies", extra={"count": len(recent_movies)})
        return self._build_collection(
            recent_movies, media_type="movie", collection_name="Recommended Movies", limit=limit
        )

    def build_show_collection(self, limit: int = 10, days: int = 30) -> List[Recommendation]:
        recent_shows = list(self.plex.recently_watched_shows(days=days, max_results=200))
        LOGGER.debug("Found recently watched shows", extra={"count": len(recent_shows)})
        return self._build_collection(
            recent_shows, media_type="tv", collection_name="Recommended Shows", limit=limit
        )
