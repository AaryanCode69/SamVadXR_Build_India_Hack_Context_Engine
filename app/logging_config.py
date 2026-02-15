"""
Structured JSON logging configuration.

All logs are JSON-formatted with standard fields required by rules.md §7.
"""

import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the root 'samvadxr' logger with JSON output to stdout.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    logger = logging.getLogger("samvadxr")

    # Avoid duplicate handlers on hot-reload
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)

    formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
        },
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
