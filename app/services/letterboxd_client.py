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
    letterboxd_keywords: Set[str]
    networks: Set[str]
    number_of_seasons: Optional[int]
    number_of_episodes: Optional[int]
    episode_run_time: Optional[int]
    status: Optional[str]
    first_air_date: Optional[str]
    tmdb_rating: Optional[float]
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
            letterboxd_keywords=set(),
            networks=set(),
            number_of_seasons=None,
            number_of_episodes=None,
            episode_run_time=None,
            status=None,
            first_air_date=None,
            tmdb_rating=None,
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

    def _fetch_letterboxd_score(
        self, title: str, year: Optional[int]
    ) -> Tuple[Optional[float], Optional[int], Optional[str], Set[str]]:
        response = self.letterboxd.get("https://letterboxd.com/search/films/", params={"q": title})
        if response.status_code == 403 or self._is_login_response(response):
            logger.warning("Letterboxd search blocked (status=%s, url=%s)", response.status_code, response.url)
            return None, None, None, set()
        if response.status_code != 200:
            return None, None, None, set()
        soup = BeautifulSoup(response.text, "html.parser")
        best_rating: Optional[float] = None
        best_votes: Optional[int] = None
        candidate_slug: Optional[str] = None
        keywords: Set[str] = set()
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
        if candidate_slug:
            page_rating, page_votes, keywords = self._fetch_film_detail(candidate_slug)
            best_rating = best_rating if best_rating is not None else page_rating
            best_votes = best_votes if best_votes is not None else page_votes
        return best_rating, best_votes, candidate_slug, keywords

    def _fetch_film_detail(self, slug: str) -> Tuple[Optional[float], Optional[int], Set[str]]:
        normalized_slug = slug if slug.startswith("/") else f"/{slug}"
        film_url = f"https://letterboxd.com{normalized_slug.rstrip('/')}/"
        response = self.letterboxd.get(film_url)
        if response.status_code == 403 or self._is_login_response(response):
            logger.warning("Letterboxd film page blocked (status=%s, url=%s)", response.status_code, response.url)
            return None, None, set()
        if response.status_code != 200:
            return None, None, set()
        soup = BeautifulSoup(response.text, "html.parser")
        rating, votes = self._extract_rating_from_film_page(soup)
        keywords = self._extract_keywords_from_film_page(soup)
        return rating, votes, keywords

    def _find_letterboxd_slug(self, title: str, year: Optional[int] = None) -> Optional[str]:
        response = self.letterboxd.get("https://letterboxd.com/search/films/", params={"q": title})
        if response.status_code == 403 or self._is_login_response(response):
            logger.warning("Letterboxd search blocked (status=%s, url=%s)", response.status_code, response.url)
            return None
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        candidate_slug: Optional[str] = None
        for li in soup.select("li[data-film-slug]"):
            try:
                film_year = int(li.get("data-film-year") or 0)
            except ValueError:
                film_year = 0
            if year and film_year and abs(film_year - year) > 1:
                continue
            candidate_slug = li.get("data-film-slug")
            if candidate_slug:
                break
        return candidate_slug

    def _fetch_letterboxd_related(self, slug: str, limit: Optional[int]) -> list[int]:
        normalized_slug = slug if slug.startswith("/") else f"/{slug}"
        film_url = f"https://letterboxd.com{normalized_slug.rstrip('/')}/"
        response = self.letterboxd.get(film_url)
        if response.status_code == 403 or self._is_login_response(response):
            logger.warning("Letterboxd related blocked (status=%s, url=%s)", response.status_code, response.url)
            return []
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        sections = soup.select(
            "section.related-films, section.films-like-this, section.also-liked, section.recommendations"
        )
        if not sections:
            sections = soup.select("section[class*='related'], section[class*='liked']")
        if not sections:
            sections = [soup]

        related_slugs: list[tuple[str, Optional[int]]] = []
        seen_slugs: set[str] = set()
        for section in sections:
            for li in section.select("li[data-film-slug]"):
                related_slug = li.get("data-film-slug")
                if not related_slug or related_slug in seen_slugs:
                    continue
                seen_slugs.add(related_slug)
                year = self._parse_int(li.get("data-film-year"))
                related_slugs.append((related_slug, year))
                if limit is not None and len(related_slugs) >= limit:
                    break
            if limit is not None and len(related_slugs) >= limit:
                break

        related_tmdb_ids: list[int] = []
        for related_slug, year in related_slugs:
            title = self._slug_to_title(related_slug)
            tmdb_id = self.search_tmdb_id(title, "movie", year)
            if tmdb_id is not None and tmdb_id not in related_tmdb_ids:
                related_tmdb_ids.append(tmdb_id)
            if limit is not None and len(related_tmdb_ids) >= limit:
                break

        return related_tmdb_ids

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
    def _extract_keywords_from_film_page(soup: BeautifulSoup) -> Set[str]:
        keywords: Set[str] = set()

        def _add_keyword(value: str) -> None:
            cleaned = value.strip()
            if cleaned:
                keywords.add(cleaned)

        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                data = json.loads(script.string or "{}")
            except json.JSONDecodeError:
                continue
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                kw_value = entry.get("keywords")
                if isinstance(kw_value, list):
                    for kw in kw_value:
                        if isinstance(kw, str):
                            _add_keyword(kw)
                elif isinstance(kw_value, str):
                    for kw in kw_value.split(","):
                        _add_keyword(kw)
        if keywords:
            return keywords

        for anchor in soup.select("a[href*='/keyword/']"):
            text = anchor.get_text(strip=True)
            if text:
                keywords.add(text)

        return keywords

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
            digits = re.sub(r"[^0-9]", "", str(value))
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
        networks = {n.get("name", "") for n in details.get("networks", []) if n.get("name")}
        number_of_seasons = self._parse_int(details.get("number_of_seasons"))
        number_of_episodes = self._parse_int(details.get("number_of_episodes"))
        first_air_date = details.get("first_air_date") or details.get("release_date")
        status = details.get("status")
        episode_run_time = details.get("episode_run_time")
        if isinstance(episode_run_time, list):
            episode_run_time = episode_run_time[0] if episode_run_time else None
        if episode_run_time is not None:
            try:
                episode_run_time = int(episode_run_time)
            except (TypeError, ValueError):
                episode_run_time = None

        keyword_entries = keywords_resp.get("keywords") or keywords_resp.get("results") or []
        keywords = {kw.get("name", "") for kw in keyword_entries if kw.get("name")}

        rating, votes, _slug, letterboxd_keywords = self._fetch_letterboxd_score(title, release_year)

        tmdb_rating = None
        try:
            vote_average = details.get("vote_average")
            tmdb_rating = float(vote_average) * 10 if vote_average is not None else None
        except (TypeError, ValueError):
            tmdb_rating = None

        return MediaProfile(
            title=title,
            tmdb_id=tmdb_id,
            media_type=media_type,
            cast=cast,
            directors=directors,
            writers=writers,
            genres=genres,
            keywords=keywords,
            letterboxd_keywords=letterboxd_keywords,
            networks=networks,
            number_of_seasons=number_of_seasons,
            number_of_episodes=number_of_episodes,
            episode_run_time=episode_run_time,
            status=status,
            first_air_date=first_air_date,
            tmdb_rating=tmdb_rating,
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

    def search_related(
        self, profile: MediaProfile, limit: int | None = 15
    ) -> List[MediaProfile]:
        """Return TMDB-similar titles for a profile (excludes recommendations)."""

        max_results = limit if limit and limit > 0 else None
        recommendations: List[MediaProfile] = []
        seen_ids: set[int] = set()

        def _fetch_candidates(path: str, remaining: Optional[int]) -> List[int]:
            results: list[int] = []
            page = 1
            while True:
                params = {"page": page}
                response = self.client.get(path, params=params).json()
                entries = response.get("results", [])
                if not entries:
                    break
                for entry in entries:
                    tmdb_id = entry.get("id")
                    if tmdb_id is None or tmdb_id in seen_ids:
                        continue
                    seen_ids.add(tmdb_id)
                    results.append(tmdb_id)
                    if remaining is not None and len(results) >= remaining:
                        break
                total_pages = response.get("total_pages") or 1
                if (remaining is not None and len(results) >= remaining) or page >= total_pages:
                    break
                page += 1
            return results

        candidate_ids: list[int] = []
        if (
            settings.letterboxd_allow_scrape
            and profile.media_type == "movie"
            and not settings.letterboxd_session
        ):
            logger.info("Letterboxd scrape enabled but no session provided; proceeding unauthenticated")

        if settings.letterboxd_allow_scrape and profile.media_type == "movie":
            slug = self._find_letterboxd_slug(profile.title)
            if slug:
                candidate_ids = self._fetch_letterboxd_related(slug, max_results)
            if not candidate_ids:
                logger.info("Falling back to TMDB similar titles for %s", profile.title)

        if not candidate_ids:
            source = f"/{profile.media_type}/{profile.tmdb_id}/similar"
            remaining = max_results
            candidate_ids = _fetch_candidates(source, remaining)
        for tmdb_id in candidate_ids:
            try:
                recommendations.append(self.fetch_profile(tmdb_id, profile.media_type))
            except Exception:
                continue
            if max_results and len(recommendations) >= max_results:
                break
        return recommendations

    @staticmethod
    def _slug_to_title(slug: str) -> str:
        cleaned = slug.strip("/").split("/")[-1]
        return cleaned.replace("-", " ") if cleaned else slug


def _tmdb_score(rating: Optional[float]) -> float:
    if rating is None:
        return 0.0
    if rating <= 50:
        return -50.0
    if rating <= 65:
        return 0.0
    if rating <= 70:
        return 10.0
    if rating <= 80:
        return 20.0
    if rating <= 90:
        return 30.0
    return 50.0


def profile_similarity(
    source: MediaProfile, target: MediaProfile
) -> tuple[float, dict[str, float]]:
    breakdown = {
        "cast": 10.0 * len(source.cast & target.cast),
        "directors": 30.0 * len(source.directors & target.directors),
        "writers": 20.0 * len(source.writers & target.writers),
        "genres": 10.0 * len(source.genres & target.genres),
        "keywords": 10.0 * len(source.keywords & target.keywords),
        "letterboxd_keywords": 15.0 * len(source.letterboxd_keywords & target.letterboxd_keywords),
        "tmdb_rating": _tmdb_score(target.tmdb_rating),
    }
    total = round(sum(breakdown.values()), 2)
    normalized_breakdown = {key: round(value, 2) for key, value in breakdown.items()}
    return total, normalized_breakdown


def get_letterboxd_client() -> LetterboxdClient:
    return LetterboxdClient()
