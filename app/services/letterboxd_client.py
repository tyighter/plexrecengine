from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import json
import httpx
import logging
import re
from bs4 import BeautifulSoup

from app.config import settings

TMDB_BASE = "https://api.themoviedb.org/3"


logger = logging.getLogger(__name__)


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
        if response.status_code == 403 or self._is_login_response(response):
            logger.warning("Letterboxd search blocked (status=%s, url=%s)", response.status_code, response.url)
            return None, None
        if response.status_code != 200:
            return None, None
        soup = BeautifulSoup(response.text, "html.parser")
        best_rating: Optional[float] = None
        best_votes: Optional[int] = None
        candidate_slug: Optional[str] = None
        for li in soup.select("li[data-film-slug]"):
            try:
                film_year = int(li.get("data-film-year") or 0)
            except ValueError:
                film_year = 0
            if year and film_year and abs(film_year - year) > 1:
                continue
            if candidate_slug is None:
                candidate_slug = li.get("data-film-slug")
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
        if best_rating is None and candidate_slug:
            logger.info("Falling back to film page scrape for '%s'", candidate_slug)
            best_rating, best_votes = self._fetch_film_detail(candidate_slug)
        return best_rating, best_votes

    def _fetch_film_detail(self, slug: str) -> Tuple[Optional[float], Optional[int]]:
        film_url = f"https://letterboxd.com{slug}/"
        response = self.letterboxd.get(film_url)
        if response.status_code == 403 or self._is_login_response(response):
            logger.warning("Letterboxd film page blocked (status=%s, url=%s)", response.status_code, response.url)
            return None, None
        if response.status_code != 200:
            return None, None
        soup = BeautifulSoup(response.text, "html.parser")
        return self._extract_rating_from_film_page(soup)

    def _extract_rating_from_film_page(self, soup: BeautifulSoup) -> Tuple[Optional[float], Optional[int]]:
        rating = None
        votes = None

        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                data = json.loads(script.string or "{}")
            except json.JSONDecodeError:
                continue
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                aggregate = entry.get("aggregateRating") if isinstance(entry, dict) else None
                if not aggregate or not isinstance(aggregate, dict):
                    continue
                rating = rating or self._parse_float(aggregate.get("ratingValue"))
                votes = votes or self._parse_int(aggregate.get("ratingCount"))
                if rating is not None:
                    break
            if rating is not None:
                break

        if rating is None:
            rating_meta = soup.find("meta", attrs={"property": "letterboxd:filmRatingAverage"})
            if rating_meta:
                rating = self._parse_float(rating_meta.get("content"))
        if votes is None:
            votes_meta = soup.find("meta", attrs={"property": "letterboxd:filmRatingCount"})
            if votes_meta:
                votes = self._parse_int(votes_meta.get("content"))

        if rating is None:
            rating_meta = soup.find("meta", attrs={"name": "twitter:data1"})
            if rating_meta:
                rating = self._parse_float(rating_meta.get("content"))
        if votes is None:
            votes_meta = soup.find("meta", attrs={"name": "twitter:data2"})
            if votes_meta:
                votes = self._parse_int(votes_meta.get("content"))

        if rating is None:
            rating_node = soup.find(attrs={"data-rating": True})
            if rating_node:
                rating = self._parse_float(rating_node.get("data-rating"))
        if votes is None:
            votes_node = soup.find(attrs={"data-rating-count": True})
            if votes_node:
                votes = self._parse_int(votes_node.get("data-rating-count"))

        return rating, votes

    @staticmethod
    def _parse_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            cleaned = value.strip().split(" ")[0].replace("★", "")
            return float(cleaned)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            digits = re.sub(r"[^0-9]", "", value)
            return int(digits) if digits else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _is_login_response(response: httpx.Response) -> bool:
        url_str = str(response.url).lower()
        return "sign-in" in url_str or "signin" in url_str or "login" in url_str

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


def profile_similarity(
    source: MediaProfile, target: MediaProfile
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "cast": 10.0 * len(source.cast & target.cast),
        "directors": 30.0 * len(source.directors & target.directors),
        "writers": 20.0 * len(source.writers & target.writers),
        "genres": 10.0 * len(source.genres & target.genres),
        "keywords": 10.0 * len(source.keywords & target.keywords),
        "letterboxd_rating": _letterboxd_score(target.letterboxd_rating),
    }
    total = round(sum(breakdown.values()), 2)
    normalized_breakdown = {key: round(value, 2) for key, value in breakdown.items()}
    return total, normalized_breakdown


def get_letterboxd_client() -> LetterboxdClient:
    return LetterboxdClient()
