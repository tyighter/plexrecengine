from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.services.plex_auth import check_login, start_login
from app.services.letterboxd_client import get_letterboxd_client
from app.services.plex_service import get_plex_service
from app.services.recommendation import RecommendationEngine

app = FastAPI(title="Plex Recommendation Engine")
templates = Jinja2Templates(directory="app/web/templates")
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")


@app.on_event("startup")
async def startup_build_collections():
    if not settings.is_plex_configured:
        return
    plex = get_plex_service()
    letterboxd = get_letterboxd_client()
    engine = RecommendationEngine(plex, letterboxd)
    engine.build_movie_collection()
    engine.build_show_collection()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if settings.is_plex_configured:
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
