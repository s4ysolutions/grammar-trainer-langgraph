#!/usr/bin/env python3
"""
Telegram bot — webhook mode.

Usage:
    WEBHOOK_URL=https://your.domain/hook uv run python -m tgbot.webhook

Requires a public URL. For local dev, use ngrok:
    ngrok http 8443
    WEBHOOK_URL=https://<ngrok-id>.ngrok.io/hook uv run python -m tgbot.webhook
"""
import os
import sys
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tgbot.handlers import build_app
from tgbot.log_config import apply_quiet_loggers


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)
    webhook_url = os.environ.get("WEBHOOK_URL")
    if not webhook_url:
        logger.error("WEBHOOK_URL not set")
        sys.exit(1)
    port = int(os.environ.get("PORT", "8443"))
    url_path = urlparse(webhook_url).path.lstrip("/")

    secret_token = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if not secret_token:
        logger.warning(
            "TELEGRAM_WEBHOOK_SECRET not set — webhook accepts requests from any source. "
            "Run: uv run python -m cli.gen_secret"
        )

    apply_quiet_loggers()
    app = build_app(token)
    logger.info("Starting webhook on port %d, path /%s...", port, url_path)
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=url_path,
        webhook_url=webhook_url,
        secret_token=secret_token,
    )


if __name__ == "__main__":
    main()
