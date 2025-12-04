from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path("/app/logs")
LOG_FILE = LOG_DIR / "generate.log"
SCORING_LOG_FILE = LOG_DIR / "scoring.log"
WEBUI_LOG_FILE = LOG_DIR / "webui.log"
COLLECTIONS_LOG_FILE = LOG_DIR / "collections.log"


_DEF_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def _configure_logger(name: str, log_file: Path, console_level: int | None = None) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(_DEF_FORMAT)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if console_level is not None:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger


def get_generate_logger() -> logging.Logger:
    """Return a shared logger for recommendation generation activity.

    The logger captures detailed debug information for the recommendation
    generation process in ``/app/logs/generate.log`` to help troubleshoot
    failures when the UI cannot reach the server.
    """

    return _configure_logger("generate", LOG_FILE, console_level=logging.INFO)


def get_scoring_logger() -> logging.Logger:
    """Return a shared logger for detailed scoring breakdowns.

    The logger writes per-movie similarity scoring data to ``/app/logs/scoring.log``
    so operators can audit why specific recommendations were selected.
    """

    return _configure_logger("scoring", SCORING_LOG_FILE)


def get_webui_logger() -> logging.Logger:
    """Return a shared logger for all web UI activity.

    Requests, rendering activity, and other web UI interactions are captured in
    ``/app/logs/webui.log`` so that slow-loading pages can be diagnosed.
    """

    return _configure_logger("webui", WEBUI_LOG_FILE, console_level=logging.INFO)


def get_collections_logger() -> logging.Logger:
    """Return a shared logger for collection creation and ordering.

    Detailed information about collection assembly, ordering decisions, and Plex
    interactions are written to ``/app/logs/collections.log`` to troubleshoot
    ordering discrepancies without duplicating scoring details.
    """

    return _configure_logger("collections", COLLECTIONS_LOG_FILE)
