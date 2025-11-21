# Static assets

This directory holds static files (images, CSS, JavaScript) served at `/static` by FastAPI. It exists even when no assets are present to prevent uvicorn/starlette from failing at startup.
