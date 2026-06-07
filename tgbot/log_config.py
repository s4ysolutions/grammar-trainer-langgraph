import logging
import os


class _FilterGetUpdates(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "getUpdates" not in record.getMessage()


def apply_quiet_loggers() -> None:
    """Suppress noisy third-party loggers when QUIET_LOGGERS=1."""
    if os.getenv("QUIET_LOGGERS") != "1":
        return
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("httpx").addFilter(_FilterGetUpdates())
