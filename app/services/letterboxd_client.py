from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup

from app.config import settings

TMDB_BASE = "https://api.themoviedb.org/3"


@dataclass
class MediaProfile:
    title: str
    tmdb_id: int
    media_type: str
    cast: Set[str]
    directors: Set[str]
    writers: Set[str]
    genres: Set[str]
    keywords: Set[str]
    letterboxd_rating: Optional[float]
    letterboxd_vote_count: Optional[int]

    @classmethod
    def empty(cls, title: str, tmdb_id: int, media_type: str) -> "MediaProfile":
        return cls(
            title=title,
            tmdb_id=tmdb_id,
            media_type=media_type,
            cast=set(),
            directors=set(),
            writers=set(),
            genres=set(),
            keywords=set(),
            letterboxd_rating=None,
            letterboxd_vote_count=None,
        )


class LetterboxdClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.tmdb_api_key
        if not self.api_key:
            raise RuntimeError("TMDB_API_KEY must be set to query metadata.")
        self.client = httpx.Client(base_url=TMDB_BASE, params={"api_key": self.api_key})
        self.letterboxd = httpx.Client(headers={"User-Agent": "plexrec/1.0"}, follow_redirects=True)
        if settings.letterboxd_session:
            self.letterboxd.cookies.set("letterboxd", settings.letterboxd_session, domain=".letterboxd.com")

    def _fetch_letterboxd_score(self, title: str, year: Optional[int]) -> Tuple[Optional[float], Optional[int]]:
        response = self.letterboxd.get("https://letterboxd.com/search/films/", params={"q": title})
        if response.status_code != 200:
            return None, None
        soup = BeautifulSoup(response.text, "html.parser")
        best_rating: Optional[float] = None
        best_votes: Optional[int] = None
        for li in soup.select("li[data-film-slug]"):
            try:
                film_year = int(li.get("data-film-year") or 0)
            except ValueError:
                film_year = 0
            if year and film_year and abs(film_year - year) > 1:
                continue
            rating_str = li.get("data-average-rating")
            votes_str = li.get("data-rating-count")
            if rating_str:
                try:
                    best_rating = float(rating_str)
                except ValueError:
                    best_rating = None
            if votes_str:
                try:
                    best_votes = int(votes_str)
                except ValueError:
                    best_votes = None
            if best_rating is not None:
                break
        return best_rating, best_votes

    def fetch_profile(self, tmdb_id: int, media_type: str) -> MediaProfile:
        details = self.client.get(f"/{media_type}/{tmdb_id}").json()
        credits = self.client.get(f"/{media_type}/{tmdb_id}/credits").json()
        keywords_resp = self.client.get(f"/{media_type}/{tmdb_id}/keywords").json()
        title = details.get("title") or details.get("name") or "Unknown"
        release_year = None
        for field in ("release_date", "first_air_date"):
            if details.get(field):
                try:
                    release_year = int(details[field].split("-")[0])
                except Exception:
                    release_year = None
                break

        cast = {member.get("name", "") for member in credits.get("cast", [])[:10] if member.get("name")}
        directors = {
            member.get("name", "")
            for member in credits.get("crew", [])
            if member.get("job", "").lower() == "director" and member.get("name")
        }
        writers = {
            member.get("name", "")
            for member in credits.get("crew", [])
            if member.get("job", "").lower() == "writer" and member.get("name")
        }
        genres = {g.get("name", "") for g in details.get("genres", []) if g.get("name")}

        keyword_entries = keywords_resp.get("keywords") or keywords_resp.get("results") or []
        keywords = {kw.get("name", "") for kw in keyword_entries if kw.get("name")}

        rating, votes = self._fetch_letterboxd_score(title, release_year)

        return MediaProfile(
            title=title,
            tmdb_id=tmdb_id,
            media_type=media_type,
            cast=cast,
            directors=directors,
            writers=writers,
            genres=genres,
            keywords=keywords,
            letterboxd_rating=rating,
            letterboxd_vote_count=votes,
        )

    def search_tmdb_id(self, title: str, media_type: str, year: Optional[int] = None) -> Optional[int]:
        params = {"query": title}
        if year:
            key = "year" if media_type == "movie" else "first_air_date_year"
            params[key] = year
        response = self.client.get(f"/search/{media_type}", params=params)
        results = response.json().get("results", [])
        for entry in results:
            tmdb_id = entry.get("id")
            if tmdb_id is not None:
                return tmdb_id
        return None

    def search_related(self, profile: MediaProfile, limit: int = 15) -> List[MediaProfile]:
        similar = self.client.get(f"/{profile.media_type}/{profile.tmdb_id}/similar").json().get("results", [])
        recommendations: List[MediaProfile] = []
        for item in similar[:limit * 2]:
            tmdb_id = item.get("id")
            if tmdb_id is None:
                continue
            try:
                recommendations.append(self.fetch_profile(tmdb_id, profile.media_type))
            except Exception:
                continue
            if len(recommendations) >= limit:
                break
        return recommendations


def _letterboxd_score(rating: Optional[float]) -> float:
    if rating is None:
        return 0.0
    if rating <= 2.0:
        return -50.0
    # Map 2.1 -> 1 and 5.0 -> 30 on a linear scale.
    slope = (30.0 - 1.0) / (5.0 - 2.1)
    return 1.0 + slope * (rating - 2.1)


def profile_similarity(source: MediaProfile, target: MediaProfile) -> float:
    score = 0.0
    score += 10.0 * len(source.cast & target.cast)
    score += 30.0 * len(source.directors & target.directors)
    score += 20.0 * len(source.writers & target.writers)
    score += 10.0 * len(source.genres & target.genres)
    score += 10.0 * len(source.keywords & target.keywords)
    score += _letterboxd_score(target.letterboxd_rating)
    return round(score, 2)


def get_letterboxd_client() -> LetterboxdClient:
    return LetterboxdClient()
