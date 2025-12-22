# Plex Recommendation Engine

A FastAPI app and lightweight dashboard that keeps two Plex collections (`Recommended Movies` and `Recommended Shows`) fresh based on what people are watching. Recommendations are generated from Plex history (via Plex directly or Tautulli), TMDB metadata, and optional Letterboxd signals. The app:

- Watches recent movie and show activity, refreshing recommendations in the background when something new is watched or added.
- Lets you choose the Plex movie/show libraries and the Plex or Tautulli user whose history should drive suggestions.
- Supports configurable recommendation size, collection ordering, and whether to include already-watched titles.
- Optionally keeps stand-up specials grouped together so they only cross-recommend with other stand-up titles.
- Serves a dashboard on port **5555** with recent activity, connection status, and poster previews.

> Plex items must include TMDB GUIDs (e.g., `tmdb://12345`) so metadata can be resolved correctly.

## Configuration

You can configure the app entirely from the dashboard (settings are persisted to `/app/config/keys.yml` and `.data/config.json`) or by supplying environment variables in a `.env` file. Key settings include:

```
PLEX_BASE_URL=http://<plex-host>:32400
PLEX_TOKEN=<plex-token>
PLEX_LIBRARY_NAMES=Movies,TV Shows      # Or set PLEX_MOVIE_LIBRARY / PLEX_SHOW_LIBRARY
PLEX_USER_ID=<optional-plex-account-id> # Filters history to a specific Plex user
TMDB_API_KEY=<tmdb-api-key>
LETTERBOXD_SESSION=<optional-letterboxd-session-cookie>
LETTERBOXD_ALLOW_SCRAPE=true            # Toggle Letterboxd scraping
RELATED_POOL_LIMIT=100                  # How many related titles to consider per item
ALLOW_WATCHED_RECOMMENDATIONS=false     # Allow already-watched titles in collections
COLLECTION_ORDER=highest_score          # random|highest_score|alphabetical|oldest_first|newest_first
STANDUP_ONLY_MATCHING=false             # When true, stand-up titles only match other stand-up titles
STANDUP_KEYWORDS=stand-up comedy        # Optional comma-separated keywords/collections to flag stand-up
DASHBOARD_TIMEOUT_SECONDS=10
RECENT_ACTIVITY_TIMEOUT_SECONDS=10
RECOMMENDATION_BUILD_TIMEOUT_SECONDS=120
# Optional Tautulli integration (history source and user filtering)
TAUTULLI_BASE_URL=https://tautulli.example.com
TAUTULLI_API_KEY=<tautulli-api-key>
TAUTULLI_USER_ID=<tautulli-user-id>
```

Stand-up detection looks at Plex genres/collections plus TMDB and Letterboxd keywords. Add custom entries to `STANDUP_KEYWORDS` (comma-separated) if your library uses a specific collection or tag to mark stand-up specials. When `STANDUP_ONLY_MATCHING` is enabled, recommendations from stand-up sources will only draw from other stand-up items, and non-stand-up sources will skip stand-up specials.

### Running with Docker Compose

```
docker compose up --build -d
```

The app will be available at `http://localhost:5555`. Mount `/app/config` (for saved tokens) and `/app/.data` (for cached configuration) if you want settings to persist between container restarts.

### Rebuilding after watches

- Configure a Plex webhook to `http://<host>:5555/webhook` to trigger an immediate refresh when something finishes playing.
- Background workers also watch recent history and recent additions, so recommendations gradually update even without the webhook.
- You can force a rebuild from the dashboard, and the cached results are shown on page load while long-running jobs continue in the background.

Similarity scoring prioritizes Letterboxd ratings (when available) layered on top of TMDB cast/crew/genre/keyword overlap. Set `LETTERBOXD_ALLOW_SCRAPE=false` to skip scraping entirely.

## Development

Install dependencies and run the server locally:

```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 5555
```
