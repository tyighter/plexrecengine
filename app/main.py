from __future__ import annotations

import asyncio
import time
from pathlib import Path

from dotenv import set_key
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl

from app.keys_store import persist_keys
from app.config import save_config, settings
from app.services.generate_logging import get_generate_logger, get_webui_logger
from app.services.plex_auth import (
    check_login,
    list_available_libraries,
    list_available_users,
    save_library_preferences,
    start_login,
)
from app.services.plex_index import get_plex_index
from app.services.plex_service import get_plex_service
from app.services.recommendation import RecommendationEngine
from app.services.tautulli_logging import log_recent_tv_activity
from app.services.tautulli_service import list_tautulli_users

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"

app = FastAPI(title="Plex Recommendation Engine")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
ENV_PATH = Path(".env")
LOGGER = get_generate_logger()
WEB_LOGGER = get_webui_logger()

RECENT_CACHE_TTL_SECONDS = 300
RECENT_ACTIVITY_CACHE: dict[str, object] = {"data": None, "timestamp": 0.0}
RECENT_CACHE_LOCK = asyncio.Lock()
LAST_SEEN_RECENT_KEYS: set[str] = set()

RECOMMENDATION_CACHE: dict[str, object] = {
    "movies": [],
    "shows": [],
    "timestamp": 0.0,
}
RECOMMENDATION_CACHE_LOCK = asyncio.Lock()
RECOMMENDATION_REBUILD_TASK: asyncio.Task | None = None
INDEX_REBUILD_TASK: asyncio.Task | None = None


@app.middleware("http")
async def log_web_requests(request: Request, call_next):
    start = time.perf_counter()
    WEB_LOGGER.info(
        "Incoming request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        WEB_LOGGER.exception(
            "Request failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    WEB_LOGGER.info(
        "Completed request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    return response


ALLOWED_COLLECTION_ORDERS = {
    "random",
    "highest_score",
    "alphabetical",
    "oldest_first",
    "newest_first",
}


class TmdbKeyRequest(BaseModel):
    apiKey: str


class PlexPreferences(BaseModel):
    movieLibrary: str
    showLibrary: str
    plexUserId: str | None = None


class TautulliConfigRequest(BaseModel):
    baseUrl: HttpUrl
    apiKey: str


class RecommendationConfig(BaseModel):
    relatedPoolLimit: int | None = None
    allowWatched: bool | None = None
    collectionOrder: str | None = None


def _serialize_recent(item, poster_url):
    def _maybe_iso(dt):
        try:
            return dt.isoformat()
        except Exception:  # noqa: BLE001
            return None

    return {
        "title": getattr(item, "title", ""),
        "poster": poster_url(item) if item else None,
        "year": getattr(item, "year", None),
        "library": getattr(item, "librarySectionTitle", None),
        "rating_key": getattr(item, "ratingKey", None),
        "last_viewed_at": _maybe_iso(getattr(item, "lastViewedAt", None)),
    }


def _is_recent_cache_fresh() -> bool:
    return (time.time() - RECENT_ACTIVITY_CACHE["timestamp"]) < RECENT_CACHE_TTL_SECONDS


async def _fetch_recent_activity() -> dict[str, list[dict[str, object]]]:
    def fetch_recent():
        plex = get_plex_service()
        try:
            log_recent_tv_activity(
                plex,
                user_id=settings.tautulli_user_id or settings.plex_user_id,
                days=7,
            )
        except Exception:
            LOGGER.exception("Failed to log recent Tautulli activity")

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
    LAST_SEEN_RECENT_KEYS.clear()


def _extract_recent_keys(data: dict[str, list[dict[str, object]]] | None) -> set[str]:
    if not data:
        return set()
    keys = set()
    for item in data.get("recent_movies", []):
        rating_key = item.get("rating_key")
        if rating_key is not None:
            keys.add(str(rating_key))
    for item in data.get("recent_shows", []):
        rating_key = item.get("rating_key")
        if rating_key is not None:
            keys.add(str(rating_key))
    return keys


def _is_rebuild_running() -> bool:
    return RECOMMENDATION_REBUILD_TASK is not None and not RECOMMENDATION_REBUILD_TASK.done()


def _is_index_rebuild_running() -> bool:
    return INDEX_REBUILD_TASK is not None and not INDEX_REBUILD_TASK.done()


async def _get_cached_recommendations() -> dict[str, object] | None:
    async with RECOMMENDATION_CACHE_LOCK:
        if RECOMMENDATION_CACHE["movies"] or RECOMMENDATION_CACHE["shows"]:
            return {
                "movies": RECOMMENDATION_CACHE["movies"],
                "shows": RECOMMENDATION_CACHE["shows"],
            }
    return None


async def _generate_recommendations(force: bool = False) -> dict[str, object]:
    async with RECOMMENDATION_CACHE_LOCK:
        if not force and RECOMMENDATION_CACHE["movies"] and RECOMMENDATION_CACHE["shows"]:
            return {
                "movies": RECOMMENDATION_CACHE["movies"],
                "shows": RECOMMENDATION_CACHE["shows"],
            }

    def build_recommendations():
        plex = get_plex_service()
        index = get_plex_index()
        engine = RecommendationEngine(plex, index)
        return engine.build_movie_collection(), engine.build_show_collection()

    movies, shows = await asyncio.to_thread(build_recommendations)
    async with RECOMMENDATION_CACHE_LOCK:
        RECOMMENDATION_CACHE["movies"] = [m.__dict__ for m in movies]
        RECOMMENDATION_CACHE["shows"] = [s.__dict__ for s in shows]
        RECOMMENDATION_CACHE["timestamp"] = time.time()
        return {"movies": RECOMMENDATION_CACHE["movies"], "shows": RECOMMENDATION_CACHE["shows"]}


def _schedule_recommendation_rebuild(force: bool = False) -> bool:
    global RECOMMENDATION_REBUILD_TASK
    if _is_rebuild_running():
        return True

    async def _run_rebuild():
        try:
            await _generate_recommendations(force=force)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Background recommendation rebuild failed")

    RECOMMENDATION_REBUILD_TASK = asyncio.create_task(_run_rebuild())
    return True


async def _run_recommendation_rebuild(force: bool = False) -> dict[str, object]:
    """Run the recommendation rebuild immediately, waiting on any in-flight run."""

    global RECOMMENDATION_REBUILD_TASK

    if _is_rebuild_running():
        await RECOMMENDATION_REBUILD_TASK  # type: ignore[arg-type]
        cached = await _get_cached_recommendations()
        if cached:
            return cached

    async def _run_and_record():
        try:
            return await _generate_recommendations(force=force)
        finally:
            RECOMMENDATION_REBUILD_TASK = None

    RECOMMENDATION_REBUILD_TASK = asyncio.create_task(_run_and_record())
    return await RECOMMENDATION_REBUILD_TASK


def _schedule_index_rebuild() -> bool:
    global INDEX_REBUILD_TASK
    if _is_index_rebuild_running():
        return True

    async def _run_rebuild():
        try:
            index = get_plex_index()
            await asyncio.to_thread(index.rebuild)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Background Plex index rebuild failed")
        finally:
            global INDEX_REBUILD_TASK
            INDEX_REBUILD_TASK = None

    INDEX_REBUILD_TASK = asyncio.create_task(_run_rebuild())
    return True


async def _watch_recent_activity():
    while True:
        try:
            if not settings.is_plex_configured:
                await asyncio.sleep(120)
                continue

            recent = await refresh_recent_cache(force=True)
            current_keys = _extract_recent_keys(recent)
            new_keys = current_keys - LAST_SEEN_RECENT_KEYS
            LAST_SEEN_RECENT_KEYS.clear()
            LAST_SEEN_RECENT_KEYS.update(current_keys)

            if new_keys:
                LOGGER.info(
                    "Detected %s new recently watched items; rebuilding recommendations",
                    len(new_keys),
                )
                _schedule_recommendation_rebuild(force=True)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Background recent activity watcher failed")
        await asyncio.sleep(120)


async def _watch_library_additions():
    while True:
        try:
            if not settings.is_plex_configured:
                await asyncio.sleep(300)
                continue

            index = get_plex_index()
            added = await asyncio.to_thread(index.refresh_recent_additions, 100)
            if added:
                LOGGER.info(
                    "Discovered %s new Plex items; refreshing recommendations and index",
                    added,
                )
                _schedule_recommendation_rebuild(force=True)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Background library addition watcher failed")
        await asyncio.sleep(300)


@app.on_event("startup")
async def startup_build_collections():
    WEB_LOGGER.info("Web UI starting up and ready to serve requests")
    if not settings.is_plex_configured:
        return

    index = get_plex_index()
    status = index.status()
    total_items = status.get("total_items") or 0
    processed_items = status.get("processed_items") or 0
    index_complete = status.get("state") == "ready" and (
        not total_items or processed_items >= total_items
    )
    if not index_complete:
        _schedule_index_rebuild()

    def build_collections():
        plex = get_plex_service()
        engine = RecommendationEngine(plex, index)
        engine.build_movie_collection()
        engine.build_show_collection()

    # Run the expensive collection refresh in the background so startup
    # doesn't block the first page load after configuration.
    asyncio.create_task(asyncio.to_thread(build_collections))
    asyncio.create_task(refresh_recent_cache())
    asyncio.create_task(_watch_recent_activity())
    asyncio.create_task(_watch_library_additions())


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    movies = []
    shows = []
    recent_movies: list[dict[str, object]] = []
    recent_shows: list[dict[str, object]] = []
    index_status: dict[str, object] = {"state": "idle"}

    if settings.is_plex_configured:
        try:
            recent = await asyncio.wait_for(
                refresh_recent_cache(), timeout=settings.recent_activity_timeout_seconds
            )
            recent_movies = list(recent.get("recent_movies", []))  # type: ignore[arg-type]
            recent_shows = list(recent.get("recent_shows", []))  # type: ignore[arg-type]
        except asyncio.TimeoutError:
            LOGGER.warning("Timed out loading recent activity for dashboard; rendering without it")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to load recent activity for dashboard")

    if settings.is_plex_configured:
        try:
            cached_recs = await asyncio.wait_for(
                _get_cached_recommendations(), timeout=settings.dashboard_timeout_seconds
            )
            if cached_recs:
                movies = cached_recs.get("movies", []) if cached_recs else []
                shows = cached_recs.get("shows", []) if cached_recs else []
            else:
                _schedule_recommendation_rebuild(force=True)
        except asyncio.TimeoutError:
            LOGGER.warning(
                "Timed out loading cached recommendations for dashboard; rendering placeholders"
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to load cached recommendations for dashboard")

        try:
            index = get_plex_index()
            index_status = index.status()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to load Plex index status")
    WEB_LOGGER.info(
        "Rendering dashboard",
        extra={
            "movies": len(movies),
            "shows": len(shows),
            "recent_movies": len(recent_movies),
            "recent_shows": len(recent_shows),
        },
    )
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "settings": settings,
            "movies": movies,
            "shows": shows,
            "recent_movies": recent_movies,
            "recent_shows": recent_shows,
            "index_status": index_status,
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
        try:
            return await asyncio.wait_for(
                refresh_recent_cache(force=True),
                timeout=settings.recent_activity_timeout_seconds,
            )
        except asyncio.TimeoutError:
            LOGGER.warning("Timed out refreshing recent activity; returning stale cache if available")
            if RECENT_ACTIVITY_CACHE["data"]:
                return RECENT_ACTIVITY_CACHE["data"]
            raise HTTPException(status_code=504, detail="Recent activity request timed out")
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


@app.get("/api/plex/users")
async def available_users():
    if not settings.is_plex_configured:
        raise HTTPException(status_code=400, detail="Plex is not configured")
    try:
        return list_available_users()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/plex/preferences")
async def update_preferences(payload: PlexPreferences):
    try:
        result = save_library_preferences(
            payload.movieLibrary, payload.showLibrary, payload.plexUserId
        )
        invalidate_recent_cache()
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/plex/index/refresh")
async def refresh_plex_index():
    if not settings.is_plex_configured:
        raise HTTPException(status_code=400, detail="Plex is not configured")
    try:
        _schedule_index_rebuild()
        index = get_plex_index()
        return index.status()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to refresh Plex index")
        raise HTTPException(status_code=500, detail="Failed to refresh Plex index") from exc


@app.get("/api/plex/index/status")
async def plex_index_status():
    if not settings.is_plex_configured:
        raise HTTPException(status_code=400, detail="Plex is not configured")
    index = get_plex_index()
    return index.status()


@app.post("/api/tautulli/config")
async def set_tautulli_config(payload: TautulliConfigRequest):
    base_url = str(payload.baseUrl).rstrip("/")
    api_key = payload.apiKey.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), "TAUTULLI_BASE_URL", base_url)
    set_key(str(ENV_PATH), "TAUTULLI_API_KEY", api_key)
    save_config({"TAUTULLI_BASE_URL": base_url, "TAUTULLI_API_KEY": api_key})
    persist_keys(tautulli_base_url=base_url, tautulli_api_key=api_key)
    settings.tautulli_base_url = base_url
    settings.tautulli_api_key = api_key
    return {"status": "ok"}


@app.get("/api/tautulli/users")
async def available_tautulli_users():
    if not settings.is_tautulli_configured:
        raise HTTPException(status_code=400, detail="Tautulli is not configured")
    try:
        return list_tautulli_users()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@app.post("/api/recommendations/config")
async def set_recommendation_config(payload: RecommendationConfig):
    if (
        payload.relatedPoolLimit is None
        and payload.allowWatched is None
        and payload.collectionOrder is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one recommendation setting to update",
        )

    if payload.relatedPoolLimit is not None:
        if payload.relatedPoolLimit < 0:
            raise HTTPException(
                status_code=400, detail="Related pool limit must be zero or greater"
            )
        ENV_PATH.touch(exist_ok=True)
        set_key(str(ENV_PATH), "RELATED_POOL_LIMIT", str(payload.relatedPoolLimit))
        save_config({"RELATED_POOL_LIMIT": payload.relatedPoolLimit})
        settings.related_pool_limit = payload.relatedPoolLimit

    if payload.allowWatched is not None:
        ENV_PATH.touch(exist_ok=True)
        set_key(
            str(ENV_PATH),
            "ALLOW_WATCHED_RECOMMENDATIONS",
            "true" if payload.allowWatched else "false",
        )
        save_config({"ALLOW_WATCHED_RECOMMENDATIONS": payload.allowWatched})
        settings.allow_watched_recommendations = payload.allowWatched

    if payload.collectionOrder is not None:
        order = payload.collectionOrder.strip().lower()
        if order not in ALLOWED_COLLECTION_ORDERS:
            raise HTTPException(status_code=400, detail="Unsupported collection order")
        ENV_PATH.touch(exist_ok=True)
        set_key(str(ENV_PATH), "COLLECTION_ORDER", order)
        save_config({"COLLECTION_ORDER": order})
        settings.collection_order = order

    return {
        "status": "ok",
        "related_pool_limit": settings.related_pool_limit,
        "allow_watched_recommendations": settings.allow_watched_recommendations,
        "collection_order": settings.collection_order,
    }


@app.post("/webhook")
async def webhook_trigger():
    if settings.is_plex_configured:
        invalidate_recent_cache()
        plex = get_plex_service()
        index = get_plex_index()
        index.refresh_recent_additions()
        engine = RecommendationEngine(plex, index)
        engine.build_movie_collection()
        engine.build_show_collection()
        _schedule_recommendation_rebuild(force=True)
        return {"status": "ok"}
    return {"status": "skipped", "reason": "Plex is not configured"}


@app.post("/api/recommendations")
async def build_recommendations():
    LOGGER.info("Received request to build recommendations")
    if not settings.is_plex_configured:
        LOGGER.warning("Plex configuration missing; rejecting recommendation request")
        raise HTTPException(status_code=400, detail="Plex is not configured")

    LOGGER.info("Rebuilding recommendations immediately")
    try:
        recommendations = await _run_recommendation_rebuild(force=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to rebuild recommendations synchronously")
        raise HTTPException(status_code=500, detail="Failed to rebuild recommendations") from exc

    LOGGER.info(
        "Returning freshly rebuilt recommendations",
        extra={
            "movies": len(recommendations.get("movies", [])),
            "shows": len(recommendations.get("shows", [])),
        },
    )
    return JSONResponse(recommendations, status_code=200)


@app.get("/api/recommendations")
async def cached_recommendations():
    if not settings.is_plex_configured:
        return {"movies": [], "shows": []}
    try:
        cached = await asyncio.wait_for(
            _get_cached_recommendations(), timeout=settings.dashboard_timeout_seconds
        )
        if cached:
            return cached
        _schedule_recommendation_rebuild(force=True)
        return {"movies": [], "shows": []}
    except asyncio.TimeoutError as exc:
        LOGGER.warning("Timed out returning cached recommendations")
        raise HTTPException(status_code=504, detail="Recommendation lookup timed out") from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to return cached recommendations")
        raise HTTPException(status_code=500, detail="Failed to load recommendations") from exc
