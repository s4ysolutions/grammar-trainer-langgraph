#!/usr/bin/env python3
"""Smoke-test the configured LLM provider/model. Exit 0 = OK, exit 1 = failed."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from agent.nodes import validate_config, _get_provider_configs, _invoke_rotating


def main():
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    print(f"Provider : {provider}")

    print("Config   : checking...")
    try:
        validate_config()
    except ValueError as e:
        print(f"FAIL     : {e}")
        sys.exit(1)
    print("Config   : OK")

    configs = _get_provider_configs()
    print(f"Chain    : {' -> '.join(c['provider'] for c in configs)}")

    print("LLM call : sending test prompt...")
    try:
        t0 = time.monotonic()
        response = _invoke_rotating("Reply with exactly one word: OK", "generator")
        elapsed = time.monotonic() - t0
    except Exception as e:
        print(f"FAIL     : {e}")
        sys.exit(1)

    print(f"LLM call : OK ({elapsed:.1f}s) — response: {response[:120]!r}")
    print("Result   : PASS")


if __name__ == "__main__":
    main()
