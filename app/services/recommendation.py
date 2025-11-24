from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from app.config import settings
from app.services.generate_logging import get_generate_logger, get_scoring_logger
from app.services.plex_index import PlexIndex, PlexProfile, profile_similarity
from app.services.plex_service import PlexService


@dataclass
class Recommendation:
    title: str
    score: float
    poster: str | None
    rating_key: int
    year: int | None = None
    source_title: str | None = None
    reason: str | None = None
    score_breakdown: dict[str, float] | None = None
    commonalities: dict[str, list[str]] | None = None


LOGGER = get_generate_logger()
SCORING_LOGGER = get_scoring_logger()


class RecommendationEngine:
    def __init__(self, plex: PlexService, index: PlexIndex) -> None:
        self.plex = plex
        self.index = index

    def _profile_for_item(self, item, media_type: str) -> Optional[PlexProfile]:
        profile = self.index.profile_for_item(item)
        if profile and profile.media_type == media_type:
            return profile
        return None

    def _score_related(
        self, source_profile: PlexProfile, candidates: Iterable[PlexProfile]
    ) -> List[Tuple[PlexProfile, float, dict[str, float]]]:
        scored = []
        plot_scores = self.index._plot_similarity_scores(source_profile)
        for candidate in candidates:
            plot_score = plot_scores.get(candidate.rating_key, 0.0)
            score, breakdown = profile_similarity(source_profile, candidate, plot_score)
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
        related_pool_limit = settings.related_pool_limit or 0
        related: list[PlexProfile] = []

        related = self.index.related_profiles(
            source_profile,
            limit=None if related_pool_limit == 0 else related_pool_limit,
        )

        if not related:
            SCORING_LOGGER.info(
                "No related Plex titles found for %s; skipping recommendations",
                source_label or "Unknown title",
            )
            return []
        else:
            SCORING_LOGGER.info(
                "Using %s Plex-related titles",
                len(related),
            )
        scored = self._score_related(source_profile, related)
        library_name = (
            settings.plex_movie_library
            if media_type == "movie"
            else settings.plex_show_library
        ) or "Plex library"

        availability: List[Tuple[PlexProfile, float, dict[str, float], list]] = []
        available_in_library = 0
        for profile, score, breakdown in scored:
            plex_item = self.plex.fetch_item(
                profile.rating_key,
                extra={"source": "related_pool"},
            )
            plex_matches = [plex_item] if plex_item else []
            if plex_item:
                available_in_library += 1
            availability.append((profile, score, breakdown, plex_matches))

        SCORING_LOGGER.info(
            "%s/%s related movies found in %s",
            available_in_library,
            len(related),
            library_name,
        )

        if not scored:
            SCORING_LOGGER.info("No related titles produced non-zero scores")
        else:
            for rank, (profile, score, breakdown, plex_matches) in enumerate(
                availability, start=1
            ):
                breakdown_text = ", ".join(
                    f"{key}={value:.2f}" for key, value in sorted(breakdown.items())
                )
                SCORING_LOGGER.info(
                    "Score #%s: %s (rating_key=%s) score=%.2f [%s]%s",
                    rank,
                    profile.title,
                    profile.rating_key,
                    score,
                    breakdown_text,
                    " — available in Plex" if plex_matches else "",
                )
        recommendations: List[Recommendation] = []
        skipped_not_in_library: list[str] = []
        for profile, score, breakdown, plex_matches in availability[: count * 2]:
            # try to find an unwatched matching item in Plex
            if not plex_matches:
                skipped_not_in_library.append(profile.title)
                SCORING_LOGGER.info(
                    "Skipping %s — not found in Plex library",
                    profile.title,
                )
                continue

            plex_item = plex_matches[0]
            if not settings.allow_watched_recommendations:
                view_count = getattr(plex_item, "viewCount", 0) or 0
                is_watched = bool(getattr(plex_item, "isWatched", False))
                if is_watched or view_count > 0:
                    SCORING_LOGGER.info(
                        "Skipping %s — already watched in Plex (viewCount=%s)",
                        profile.title,
                        view_count,
                    )
                    continue
            overlap = {
                "directors": sorted(source_profile.directors & profile.directors),
                "writers": sorted(source_profile.writers & profile.writers),
                "cast": sorted(source_profile.cast & profile.cast),
                "genres": sorted(source_profile.genres & profile.genres),
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
            reason_parts.append(f"It pairs well with {profile.title}")

            shared_traits = [
                describe(overlap["directors"], "shares director"),
                describe(overlap["cast"], "features"),
                describe(overlap["writers"], "writer"),
                describe(overlap["genres"], "genre"),
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
                    year=getattr(plex_item, "year", None),
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
            top_summary = ", ".join(
                f"#{idx} {rec.title} (score={rec.score:.2f})"
                for idx, rec in enumerate(top_selected, start=1)
            )
            SCORING_LOGGER.info(
                "Top %s selected for the collection: %s",
                len(top_selected),
                top_summary,
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
        final = self._order_recommendations(final)
        LOGGER.debug(
            "Compiled %s recommendations", collection_name,
            extra={"sources": len(source_order), "final": len(final)},
        )
        self._refresh_collection(collection_name, final)
        return final

    def _order_recommendations(self, recs: List[Recommendation]) -> List[Recommendation]:
        order = (settings.collection_order or "highest_score").lower()
        ordered = list(recs)

        if order == "random":
            random.shuffle(ordered)
            return ordered
        if order == "alphabetical":
            return sorted(ordered, key=lambda rec: rec.title or "")
        if order == "oldest_first":
            return sorted(ordered, key=lambda rec: ((rec.year or 9999), rec.title or ""))
        if order == "newest_first":
            return sorted(ordered, key=lambda rec: (-(rec.year or 0), rec.title or ""))

        return sorted(ordered, key=lambda rec: rec.score, reverse=True)

    def _order_collection_items(
        self, items: List[tuple[Recommendation, object]]
    ) -> List[object]:
        order = (settings.collection_order or "highest_score").lower()
        ordered = list(items)

        def _year(rec: Recommendation, item: object):
            if rec.year:
                return rec.year
            return getattr(item, "year", None) or getattr(
                getattr(item, "originallyAvailableAt", None), "year", 9999
            )

        if order == "random":
            random.shuffle(ordered)
        elif order == "alphabetical":
            ordered.sort(key=lambda pair: (pair[0].title or getattr(pair[1], "title", "")))
        elif order == "oldest_first":
            ordered.sort(key=lambda pair: (_year(pair[0], pair[1]) or 9999, pair[0].title or ""))
        elif order == "newest_first":
            ordered.sort(key=lambda pair: (-(
                _year(pair[0], pair[1]) or 0
            ), pair[0].title or ""))
        else:
            ordered.sort(key=lambda pair: pair[0].score, reverse=True)

        return [item for _, item in ordered]

    def _refresh_collection(self, collection_name: str, recs: List[Recommendation]):
        items: List[tuple[Recommendation, object]] = []
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
                items.append((rec, item))
        try:
            ordered_items = self._order_collection_items(items)
            self.plex.set_collection_members(collection_name, ordered_items)
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
