# Plex Recommendation Engine

A containerized FastAPI app that builds Plex collections based on your recently watched movies and shows using TMDB/Letterboxd-like metadata. On startup (and whenever the webhook is invoked), the app:

- Looks at your 10 most recently watched movies and fetches unwatched, related titles based on cast, crew, keywords, and genres.
- Looks at your 10 most recently watched shows (one entry per show) and recommends unwatched shows.
- Updates two Plex collections: `Recommended Movies` and `Recommended Shows`.
- Serves a simple web UI on port **5555** that displays the recommended posters.

## Configuration

Create a `.env` file with your Plex and metadata keys:

```
PLEX_BASE_URL=http://<your-plex-host>:32400
PLEX_TOKEN=<plex-token>
PLEX_LIBRARY_NAMES=Movies,TV Shows
TMDB_API_KEY=<tmdb-api-key>
LETTERBOXD_SESSION=<optional-letterboxd-session-cookie>
```

> The recommendation engine expects Plex GUIDs to include TMDB IDs (e.g., `tmdb://12345`).

## Run with Docker Compose

```
docker compose up --build -d
```

The app will be available at `http://localhost:5555`.

### Rebuilding after watches

Configure a Plex webhook to `http://<host>:5555/webhook` so the collections refresh whenever something is watched. You can also manually refresh by loading the dashboard.

Similarity scoring prioritizes Letterboxd data—higher rated films/shows receive a boost on top of cast/crew/genre/keyword overlap. Ratings are scraped from Letterboxd search results; providing a `LETTERBOXD_SESSION` cookie can improve reliability if Letterboxd requires sign-in.

## Development

Install dependencies and run the server locally:

```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 5555
```
