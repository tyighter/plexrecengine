from __future__ import annotations

import asyncio
import time
from pathlib import Path

from dotenv import set_key
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.keys_store import persist_keys
from app.config import save_config, settings
from app.services.generate_logging import get_generate_logger
from app.services.plex_auth import (
    check_login,
    list_available_libraries,
    save_library_preferences,
    start_login,
)
from app.services.letterboxd_client import get_letterboxd_client
from app.services.plex_service import get_plex_service
from app.services.recommendation import RecommendationEngine

app = FastAPI(title="Plex Recommendation Engine")
templates = Jinja2Templates(directory="app/web/templates")
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
ENV_PATH = Path(".env")
LOGGER = get_generate_logger()

RECENT_CACHE_TTL_SECONDS = 300
RECENT_ACTIVITY_CACHE: dict[str, object] = {"data": None, "timestamp": 0.0}
RECENT_CACHE_LOCK = asyncio.Lock()


class TmdbKeyRequest(BaseModel):
    apiKey: str


class LibrarySelection(BaseModel):
    movieLibrary: str
    showLibrary: str


def _serialize_recent(item, poster_url):
    return {
        "title": getattr(item, "title", ""),
        "poster": poster_url(item) if item else None,
        "year": getattr(item, "year", None),
        "library": getattr(item, "librarySectionTitle", None),
    }


def _is_recent_cache_fresh() -> bool:
    return (time.time() - RECENT_ACTIVITY_CACHE["timestamp"]) < RECENT_CACHE_TTL_SECONDS


async def _fetch_recent_activity() -> dict[str, list[dict[str, object]]]:
    def fetch_recent():
        plex = get_plex_service()
        return {
            "recent_movies": [
                _serialize_recent(item, plex.poster_url)
                for item in plex.recently_watched_movies(days=30, max_results=200)
            ],
            "recent_shows": [
                _serialize_recent(item, plex.poster_url)
                for item in plex.recently_watched_shows(days=30, max_results=200)
            ],
        }

    return await asyncio.to_thread(fetch_recent)


async def refresh_recent_cache(force: bool = False) -> dict[str, list[dict[str, object]]]:
    async with RECENT_CACHE_LOCK:
        if not force and RECENT_ACTIVITY_CACHE["data"] and _is_recent_cache_fresh():
            return RECENT_ACTIVITY_CACHE["data"]  # type: ignore[return-value]

        data = await _fetch_recent_activity()
        RECENT_ACTIVITY_CACHE["data"] = data
        RECENT_ACTIVITY_CACHE["timestamp"] = time.time()
        return data


def invalidate_recent_cache() -> None:
    RECENT_ACTIVITY_CACHE["timestamp"] = 0.0
    RECENT_ACTIVITY_CACHE["data"] = None


@app.on_event("startup")
async def startup_build_collections():
    if not settings.is_plex_configured or not settings.tmdb_api_key:
        return

    def build_collections():
        plex = get_plex_service()
        letterboxd = get_letterboxd_client()
        engine = RecommendationEngine(plex, letterboxd)
        engine.build_movie_collection()
        engine.build_show_collection()

    # Run the expensive collection refresh in the background so startup
    # doesn't block the first page load after configuration.
    asyncio.create_task(asyncio.to_thread(build_collections))
    if settings.is_plex_configured:
        asyncio.create_task(refresh_recent_cache())


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    movies = []
    shows = []
    recent_movies = []
    recent_shows = []
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "settings": settings,
            "movies": movies,
            "shows": shows,
            "recent_movies": recent_movies,
            "recent_shows": recent_shows,
        },
    )


@app.get("/api/recent")
async def recent_activity():
    """Return recent watch activity without blocking the main thread."""

    if not settings.is_plex_configured:
        return {"recent_movies": [], "recent_shows": []}

    if RECENT_ACTIVITY_CACHE["data"] and _is_recent_cache_fresh():
        return RECENT_ACTIVITY_CACHE["data"]

    try:
        return await refresh_recent_cache(force=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to refresh recent activity; returning stale cache if available")
        if RECENT_ACTIVITY_CACHE["data"]:
            return RECENT_ACTIVITY_CACHE["data"]
        raise HTTPException(status_code=500, detail="Failed to fetch recent activity") from exc


@app.post("/api/plex/login/start")
async def start_plex_login():
    return start_login()


@app.get("/api/plex/login/status")
async def plex_login_status(pinId: str = Query(..., alias="pinId")):
    status = check_login(pinId)
    return status.dict()


@app.get("/api/plex/libraries")
async def available_libraries():
    if not settings.is_plex_configured:
        raise HTTPException(status_code=400, detail="Plex is not configured")
    try:
        return list_available_libraries()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/plex/libraries")
async def update_libraries(payload: LibrarySelection):
    try:
        return save_library_preferences(payload.movieLibrary, payload.showLibrary)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/tmdb/key")
async def set_tmdb_api_key(payload: TmdbKeyRequest):
    api_key = payload.apiKey.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "TMDB_API_KEY", api_key)
    save_config({"TMDB_API_KEY": api_key})
    persist_keys(tmdb_api_key=api_key)
    settings.tmdb_api_key = api_key
    return {"status": "ok"}


@app.post("/webhook")
async def webhook_trigger():
    if settings.is_plex_configured:
        invalidate_recent_cache()
        plex = get_plex_service()
        letterboxd = get_letterboxd_client()
        engine = RecommendationEngine(plex, letterboxd)
        engine.build_movie_collection()
        engine.build_show_collection()
        return {"status": "ok"}
    return {"status": "skipped", "reason": "Plex is not configured"}


@app.post("/api/recommendations")
async def build_recommendations():
    LOGGER.info("Received request to build recommendations")
    if not settings.is_plex_configured:
        LOGGER.warning("Plex configuration missing; rejecting recommendation request")
        raise HTTPException(status_code=400, detail="Plex is not configured")
    if not settings.tmdb_api_key:
        LOGGER.warning("TMDB API key missing; rejecting recommendation request")
        raise HTTPException(status_code=400, detail="TMDB API key is missing")

    try:
        plex = get_plex_service()
        letterboxd = get_letterboxd_client()
        LOGGER.debug("Initialized Plex and Letterboxd clients")

        engine = RecommendationEngine(plex, letterboxd)
        LOGGER.debug("Starting movie recommendations")
        movies = engine.build_movie_collection()
        LOGGER.debug("Starting show recommendations")
        shows = engine.build_show_collection()
        LOGGER.info("Recommendation generation complete", extra={"movies": len(movies), "shows": len(shows)})
        return {
            "movies": [m.__dict__ for m in movies],
            "shows": [s.__dict__ for s in shows],
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unexpected error during recommendation generation")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations") from exc
