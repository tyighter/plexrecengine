from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

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
    crew: Set[str]
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
            crew=set(),
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
        crew = {member.get("name", "") for member in credits.get("crew", []) if member.get("department") in {"Directing", "Writing"}}
        genres = {g.get("name", "") for g in details.get("genres", []) if g.get("name")}

        keyword_entries = keywords_resp.get("keywords") or keywords_resp.get("results") or []
        keywords = {kw.get("name", "") for kw in keyword_entries if kw.get("name")}

        rating, votes = self._fetch_letterboxd_score(title, release_year)

        return MediaProfile(
            title=title,
            tmdb_id=tmdb_id,
            media_type=media_type,
            cast=cast,
            crew=crew,
            genres=genres,
            keywords=keywords,
            letterboxd_rating=rating,
            letterboxd_vote_count=votes,
        )

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


def profile_similarity(source: MediaProfile, target: MediaProfile) -> float:
    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0

    weights: Dict[str, float] = {
        "cast": 0.3,
        "crew": 0.2,
        "genres": 0.2,
        "keywords": 0.1,
        "rating": 0.2,
    }

    score = 0.0
    score += weights["cast"] * jaccard(source.cast, target.cast)
    score += weights["crew"] * jaccard(source.crew, target.crew)
    score += weights["genres"] * jaccard(source.genres, target.genres)
    score += weights["keywords"] * jaccard(source.keywords, target.keywords)
    if target.letterboxd_rating:
        score += weights["rating"] * (target.letterboxd_rating / 5.0)
    return round(score, 4)


def get_letterboxd_client() -> LetterboxdClient:
    return LetterboxdClient()
