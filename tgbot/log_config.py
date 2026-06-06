import logging
import os


def apply_quiet_loggers() -> None:
    """Suppress noisy third-party loggers when QUIET_LOGGERS=1."""
    if os.getenv("QUIET_LOGGERS") != "1":
        return
    for name in (
        "google_genai.models",  # AFC is enabled with max remote calls: 10
        "httpx",                # getUpdates / all HTTP traffic
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
