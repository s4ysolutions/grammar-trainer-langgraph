#!/usr/bin/env python3
"""
Telegram bot — polling mode.

Usage:
    uv run python -m tgbot.polling
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tgbot.handlers import build_app


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)
    app = build_app(token)
    logger.info("Starting polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
