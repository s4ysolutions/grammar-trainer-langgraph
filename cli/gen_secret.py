#!/usr/bin/env python3
"""Generate a cryptographically secure webhook secret and write it to .env."""
import re
import secrets
import sys
from pathlib import Path

_ENV_FILE = Path(__file__).parent.parent / ".env"
_KEY = "TELEGRAM_WEBHOOK_SECRET"


def main():
    token = secrets.token_urlsafe(32)

    if not _ENV_FILE.exists():
        print(f"{_KEY}={token}")
        return

    text = _ENV_FILE.read_text()
    pattern = re.compile(rf"^{_KEY}=.*$", re.MULTILINE)

    if pattern.search(text):
        updated = pattern.sub(f"{_KEY}={token}", text)
        _ENV_FILE.write_text(updated)
        print(f"Updated {_KEY} in {_ENV_FILE}", file=sys.stderr)
    else:
        _ENV_FILE.write_text(text.rstrip("\n") + f"\n{_KEY}={token}\n")
        print(f"Appended {_KEY} to {_ENV_FILE}", file=sys.stderr)

    print(token)


if __name__ == "__main__":
    main()
