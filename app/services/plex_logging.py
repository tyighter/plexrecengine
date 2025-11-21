from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path("/app/logs")
LOG_FILE = LOG_DIR / "plex.log"


_DEF_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def get_plex_logger() -> logging.Logger:
    """Return a shared logger for all Plex-related operations.

    The logger writes verbose debug information to ``/app/logs/plex.log`` while
    still emitting informational messages to the console for quick visibility.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.touch(exist_ok=True)

    logger = logging.getLogger("plex")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(_DEF_FORMAT)

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
