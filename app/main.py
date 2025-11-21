from __future__ import annotations

from pathlib import Path

from dotenv import set_key
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import settings
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


class TmdbKeyRequest(BaseModel):
    apiKey: str


class LibrarySelection(BaseModel):
    movieLibrary: str
    showLibrary: str


@app.on_event("startup")
async def startup_build_collections():
    if not settings.is_plex_configured or not settings.tmdb_api_key:
        return
    plex = get_plex_service()
    letterboxd = get_letterboxd_client()
    engine = RecommendationEngine(plex, letterboxd)
    engine.build_movie_collection()
    engine.build_show_collection()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if settings.is_plex_configured and settings.tmdb_api_key:
        plex = get_plex_service()
        letterboxd = get_letterboxd_client()
        engine = RecommendationEngine(plex, letterboxd)
        movies = engine.build_movie_collection()
        shows = engine.build_show_collection()
    else:
        movies = []
        shows = []
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "settings": settings,
            "movies": movies,
            "shows": shows,
        },
    )


@app.post("/api/plex/login/start")
async def start_plex_login():
    return start_login()


@app.get("/api/plex/login/status")
async def plex_login_status(pinId: str = Query(..., alias="pinId")):
    status = check_login(pinId)
    if status.status == "invalid":
        raise HTTPException(status_code=404, detail="Invalid PIN identifier")
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
    settings.tmdb_api_key = api_key
    return {"status": "ok"}


@app.post("/webhook")
async def webhook_trigger():
    if settings.is_plex_configured:
        plex = get_plex_service()
        letterboxd = get_letterboxd_client()
        engine = RecommendationEngine(plex, letterboxd)
        engine.build_movie_collection()
        engine.build_show_collection()
        return {"status": "ok"}
    return {"status": "skipped", "reason": "Plex is not configured"}
