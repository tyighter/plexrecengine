from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path("/app/logs")
LOG_FILE = LOG_DIR / "generate.log"


_DEF_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def get_generate_logger() -> logging.Logger:
    """Return a shared logger for recommendation generation activity.

    The logger captures detailed debug information for the recommendation
    generation process in ``/app/logs/generate.log`` to help troubleshoot
    failures when the UI cannot reach the server.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.touch(exist_ok=True)

    logger = logging.getLogger("generate")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(_DEF_FORMAT)

        # Open the log file in write mode so each container start begins with a
        # fresh generate log.
        file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
